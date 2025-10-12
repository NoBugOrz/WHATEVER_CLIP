import torch
from models.clip import clip
from torch import nn
import json
from models.clip.clip import tokenize
from ultralytics import cfg


class SpatioTemporalFeatureExtractor(nn.Module):
    def __init__(self, feature_dim, hidden_dim=None, num_heads=4, dropout=0.1):
        """
        参数:
            feature_dim: 输入特征维度
            hidden_dim: 隐藏层维度，默认与feature_dim相同
            num_heads: 多头注意力的头数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim

        self.time_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feature_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 空间注意力 - 使用多头自注意力捕捉特征间的关联
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        # 残差连接的层归一化
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # 最终输出投影
        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征，形状为[batchsize, num_frames, feature_dim]

        返回:
            输出特征，形状为[batchsize, num_frames, feature_dim]
        """
        # 保存残差连接
        residual = x

        # 时间卷积需要调整维度 [batch, num_frames, feature_dim] -> [batch, feature_dim, num_frames]
        x_time = x.permute(0, 2, 1)
        x_time = self.time_conv(x_time)
        # 转换回原来的维度 [batch, num_frames, hidden_dim]
        x = x_time.permute(0, 2, 1)

        # 残差连接 + 层归一化
        x = self.norm1(x + residual)

        # 保存残差连接
        residual = x

        # 自注意力机制
        attn_output, _ = self.attention(x, x, x)
        x = attn_output

        # 残差连接 + 层归一化
        x = self.norm2(x + residual)

        # 保存残差连接
        residual = x

        # 前馈网络
        x = self.feed_forward(x)

        # 残差连接
        x = x + residual

        # 投影回原始特征维度
        x = self.output_proj(x)

        return x



class xxx_clip(nn.Module):
    def __init__(self, config, device, is_teacher:bool):
        super().__init__()
        self.device = device
        self.config = config
        self.clip = get_clip(config,is_teacher=is_teacher)
        self.class_names = load_class_names(config.DATA.CLASS_NAMES)
        self.is_teacher = is_teacher
        self.init_model()
        self.output_dim = self.clip.visual.output_dim

    def init_model(self):
        self.text_encoder = TextEncoder(self.config, self.class_names, self.clip, self.device)
        self.image_encoder = ImageEncoder(self.config, self.clip, self.device, self.is_teacher)
        if self.is_teacher:
            for k,v in self.text_encoder.named_parameters():
                v.requires_grad = False
            for k,v in self.image_encoder.named_parameters():
                v.requires_grad = False
        return None

    @property
    def dtype(self):
        return self.clip.dtype

    def encode_image(self, img):
        return self.image_encoder(img)

    def forward(self, img):
        '''
        Args:
            img: torch tensor,shape [B*n_f, 3, H, W]
        Returns:
            clip logits
        '''
        image_features = self.image_encoder(img)
        text_features, logits = self.text_encoder(image_features)
        return image_features, text_features, logits


class TextEncoder(nn.Module):
    def __init__(self, config, class_names, clip_model, device):
        super().__init__()
        self.device = device
        self.clip_model = clip_model
        self.class_names = class_names # list, len=num class_names
        self.dtype = clip_model.dtype
        self.tokens = self._tokenize(self.class_names) # list, len=num class_names    tokens[0]: tensor(1,77)
        self._tokens = torch.stack(self.tokens).squeeze(1).to(device)
        self.short_cut = self._forward(self._tokens) # tensor, shape=[num_classes, 512]


    def _tokenize(self, class_names):
        prompted_class_names = [ "In this picture, a person is" + name for name in class_names]
        tokens = [tokenize(name) for name in prompted_class_names]
        return tokens

    def forward(self, image_features):
        '''image_features already normalized'''
        text_features = self._forward(self._tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits = (100.0 * image_features @ text_features.t()).softmax(dim=-1)
        logits = image_features @ text_features.t()
        logits = 100. * logits
        logits = logits.softmax(dim=-1)

        return text_features, logits

    def _forward(self, text):
        x = self.clip_model.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

        return x



class ImageEncoder(nn.Module):
    def __init__(self, config, clip_model, device, is_teacher:bool):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.output_dim = self.clip_model.visual.output_dim
        self.device = device
        # self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_frames = config.DATA.NUM_FRAMES
        self.linear = nn.Linear(self.output_dim, self.output_dim).to(self.device).to(torch.float16)
        self.num_frames = config.DATA.NUM_FRAMES
        self.is_teacher = is_teacher


    def forward(self, x):
        '''
        Args:
            x: raw image, tensor shape [bz * num_frames, 3, H, W]
        Returns:
            either through mean pooling or modules to be built
            video_encode: tensor shape [bz, output_dim] if student model
                          tensor shape [bz * num_frames, output_dim] if teacher model
        '''
        video_encode = self.clip_model.encode_image(x)
        video_encode = video_encode / video_encode.norm(dim=-1, keepdim=True)
        video_encode = video_encode.reshape(-1, self.num_frames, self.output_dim) # shape = [bz, num_frames, output_dim]
        if self.is_teacher:
            '''教师模型应该是原始clip'''
            return video_encode
        #TODO 直接mean_pooling效果不行，找一个可以结合时空信息的模块
        video_encode = video_encode.mean(dim=1)
        return video_encode


def get_clip(cfg,is_teacher=False):
    '''
    Returns: raw clip with parameters frozen if is_teacher is True
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(cfg.MODEL.ARCH, device=device)
    if is_teacher:
        for k,v in clip_model.named_parameters():
            v.requires_grad = False
    return clip_model

def load_class_names(json_file):
    '''
    Returns: a list of class names
    '''
    with open(json_file, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return [v for k,v in data_dict.items()]