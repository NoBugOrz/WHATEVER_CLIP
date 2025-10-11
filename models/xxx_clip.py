import torch
from models.clip import clip
from torch import nn
import json
from models.clip.clip import tokenize
from ultralytics import cfg


class multi_level_conv(nn.Module):
    def __init__(self,input_dim,output_dim,device):
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim


    def init_convs(self):
        pass
    
    def forward(self, x):
        pass



class xxx_clip(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.clip = get_clip(config,is_teacher=False)
        self.class_names = load_class_names(config.DATA.CLASS_NAMES)
        self.text_encoder = TextEncoder(config, self.class_names, self.clip, self.device)
        self.image_encoder = ImageEncoder(config, self.clip, self.device)

    @property
    def dtype(self):
        return self.clip.dtype

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
        '''image_features haven't been normalized'''
        text_features = self._forward(self._tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.t()).softmax(dim=-1)

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
    def __init__(self, config, clip_model, device):
        super().__init__()
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.output_dim = self.clip_model.visual.output_dim
        self.device = device
        # self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_frames = config.DATA.NUM_FRAMES
        self.linear = nn.Linear(self.output_dim, self.output_dim).to(self.device).to(torch.float16)
        self.num_frames = config.DATA.NUM_FRAMES


    def forward(self, x):
        '''
        Args:
            x: raw image, tensor shape [bz * num_frames, 3, H, W]
        Returns:
            either through mean pooling or modules to be built
            video_encode: tensor shape [bz, output_dim]  (not yet normalized)
        '''
        video_encode = self.clip_model.encode_image(x)
        video_encode = video_encode.reshape(-1, self.num_frames, self.output_dim) # shape = [bz, num_frames, output_dim]

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