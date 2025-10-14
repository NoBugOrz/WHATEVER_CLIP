import torch
from models.clip import clip
from torch import nn
import json
from models.clip.clip import tokenize
from ultralytics import cfg
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_, constant_

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class AttentionBlock(nn.Module):
    """
    single attention block with
    {
    self_attention,
    bottle_neck feed forward,(1 -> 2 -> 1)
    residual connection
    }
    """

    def __init__(self, hidden_dim, num_heads, device, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        ).to(device)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ).to(device)
        self.norm1 = nn.LayerNorm(hidden_dim).to(device)  # 注意力后归一化
        self.norm2 = nn.LayerNorm(hidden_dim).to(device)  # 前馈网络后归一化
        self.dropout1 = nn.Dropout(dropout).to(device)  # 注意力输出dropout
        self.dropout2 = nn.Dropout(dropout).to(device)  # 前馈网络输出dropout

        # 初始化当前块参数
        self._initialize_block()

    def _initialize_block(self):
        # 注意力层初始化
        for name, param in self.attention.named_parameters():
            if 'weight' in name:
                xavier_normal_(param, gain=0.02)
            elif 'bias' in name:
                constant_(param, 0.0)

        # 前馈网络初始化
        for m in self.feed_forward:
            if isinstance(m, nn.Linear):
                xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    constant_(m.bias, 0.0)

        # 归一化层初始化
        for m in [self.norm1, self.norm2]:
            constant_(m.weight, 1.0)
            constant_(m.bias, 0.0)

    def forward(self, x):
        # multi-attention
        attn_output, _ = self.attention(x, x, x)
        # residual connection and layer norm
        x = self.norm1(x + self.dropout1(attn_output))

        # feed_forward
        ff_output = self.feed_forward(x)
        # residual connection and layer norm
        x = self.norm2(x + self.dropout2(ff_output))

        return x

class SpatioTemporalAggregator(nn.Module):
    def __init__(self, feature_dim, num_attention_blocks, device='cuda', hidden_dim=None, num_heads=4, dropout=0.1):
        """
            feature_dim: 输入输出特征维度
            hidden_dim: 隐藏层维度，默认与feature_dim相同
            num_heads: 多头注意力的头数
            dropout: dropout概率
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim
        self.device = device
        self.num_heads = num_heads
        self.num_attention_blocks = num_attention_blocks
        self.st_feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feature_dim,
                out_channels=self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            QuickGELU(),
            nn.Dropout(dropout)

            # nn.Conv1d(
            #     in_channels=self.hidden_dim,
            #     out_channels=self.hidden_dim,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1
            # ),
            # QuickGELU(),
            # nn.Dropout(dropout)
        ).to(device)

        self.post_conv_norm = nn.LayerNorm(self.hidden_dim).to(device)

        # aggregate temporal information
        self.time_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            QuickGELU(),
            nn.Linear(self.hidden_dim // 2, 1)
        ).to(device)

        # multi-head attention
        self.attention = nn.Sequential(
            *[AttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                device=self.device,
                dropout=0.1
            ) for _ in range(self.num_attention_blocks)]
        )

        self.output_proj = nn.Linear(self.hidden_dim, self.feature_dim).to(device)

        self.norm = nn.LayerNorm(self.hidden_dim).to(device)

        self._initialize_weights()

    def _initialize_weights(self):
        # 1. 卷积层初始化 - 使用Kaiming初始化（适合ReLU激活）
        for m in self.st_feature_extractor:
            if isinstance(m, nn.Conv1d):
                # 针对ReLU激活的Kaiming正态分布初始化
                kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    constant_(m.bias, 0.0)  # 偏置初始化为0

        # 2. 注意力池化层初始化
        for m in self.time_attention:
            if isinstance(m, nn.Linear):
                # 针对Tanh激活使用Xavier初始化
                xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    constant_(m.bias, 0.0)

        # # 3. 多头注意力参数初始化
        # # 对查询/键/值投影矩阵使用特殊初始化
        # for block in self.attention:
        #     for name, param in block.named_parameters():
        #         if 'weight' in name:
        #             # 注意力机制权重使用较小的初始化范围
        #             xavier_normal_(param, gain=0.02)
        #         elif 'bias' in name:
        #             constant_(param, 0.0)

        # 4. 输出投影层初始化
        if isinstance(self.output_proj, nn.Linear):
            # 保持输出尺度与输入一致
            xavier_normal_(self.output_proj.weight, gain=1.0)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias, 0.0)

        # 5. LayerNorm初始化 - 权重为1，偏置为0（保证初始不改变输入）
        for m in [self.post_conv_norm, self.norm]:
            if isinstance(m, nn.LayerNorm):
                constant_(m.weight, 1.0)
                constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: shape [batchsize, num_frames, feature_dim]

        global_feature: shape [batchsize, feature_dim]
        """
        batch_size, num_frames, _ = x.shape

        # [batch, num_frames, feature_dim] -> [batch, feature_dim, num_frames]
        x_time = x.permute(0, 2, 1).to(torch.float32)
        x_time = self.st_feature_extractor(x_time)
        # [batch, num_frames, hidden_dim]
        x = x_time.permute(0, 2, 1)
        x = self.post_conv_norm(x)

        attn_output = self.attention(x)
        x = self.norm(x + attn_output)  # residual

        # 3. 时间维度聚合 - 注意力加权池化
        # 计算每个时间步的注意力权重
        attn_weights = self.time_attention(x)  # [batch, num_frames, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # 归一化权重

        # 加权求和得到最终特征 [batch, hidden_dim]
        global_feature = torch.sum(x * attn_weights, dim=1)

        # project to output dim
        global_feature = self.output_proj(global_feature)

        return global_feature.to(torch.float32)



class xxx_clip(nn.Module):
    def __init__(self, config, device, is_teacher:bool):
        super().__init__()
        self.device = device
        self.config = config
        self.clip = get_clip(config,is_teacher=is_teacher)
        self.class_names = load_class_names(config.DATA.CLASS_NAMES)
        self.is_teacher = is_teacher
        self.output_dim = self.clip.visual.output_dim
        self.init_model()


    def init_model(self):
        self.text_encoder = TextEncoder(self.config, self.class_names, self.clip, self.device)
        self.image_encoder = ImageEncoder(self.config, self.clip, self.device, self.is_teacher)
        if self.is_teacher:
            for k,v in self.text_encoder.named_parameters():
                v.requires_grad = False
            for k,v in self.image_encoder.named_parameters():
                v.requires_grad = False

        if not self.is_teacher:
            self.spatial_temporal_module = SpatioTemporalAggregator(feature_dim=self.output_dim,
                                                                    num_attention_blocks=1,
                                                                    device=self.device,
                                                                    hidden_dim=self.output_dim,
                                                                    num_heads=4,
                                                                    dropout=0.1
                                                                    )
            for k,v in self.text_encoder.named_parameters():
                if '11' in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False
            for k,v in self.image_encoder.named_parameters():
                if '11' in k:
                    v.requires_grad = True
                else:
                    v.requires_grad = False


    @property
    def dtype(self):
        return self.clip.dtype

    def encode_image(self, img):
        image_encode = self.image_encoder(img) # shape[bz, n_f, feat_dim]
        '''
        if teacher model, use mean-pooling return shape: [bz, feat_dim]
                                             else shape: [bz, n_f, feat_dim]
        '''
        if self.is_teacher:
            image_encode = image_encode.mean(dim=1) # shape [bz, feat_dim]
            return image_encode
        else:
            return image_encode # shape[bz, n_f, feat_dim]

    def forward(self, img):
        '''
        Args:
            img: torch tensor,shape [B*n_f, 3, H, W]
        Returns:
            clip logits
        '''
        image_features = self.image_encoder(img) # shape[bz, n_f, feat_dim]

        if self.is_teacher:
            text_features, logits = self.text_encoder(image_features)
            return image_features, text_features, logits

        image_features = self.spatial_temporal_module(image_features)
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

        return x.to(torch.float32)



class ImageEncoder(nn.Module):
    def __init__(self, config, clip_model, device, is_teacher:bool):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.output_dim = self.clip_model.visual.output_dim
        self.device = device
        # self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_frames = config.DATA.NUM_FRAMES
        self.linear = nn.Linear(self.output_dim, self.output_dim).to(self.device).to(torch.float32)
        self.num_frames = config.DATA.NUM_FRAMES
        self.is_teacher = is_teacher


    def forward(self, x):
        '''
        Args:
            x: raw image, tensor shape [bz * num_frames, 3, H, W]
        Returns:
            either through mean pooling or modules to be built
            video_encode: tensor shape [bz, num_frames, output_dim]
        '''
        video_encode = self.clip_model.encode_image(x)
        # video_encode = video_encode / video_encode.norm(dim=-1, keepdim=True)
        video_encode = video_encode.reshape(-1, self.num_frames, self.output_dim) # shape = [bz, num_frames, output_dim]
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