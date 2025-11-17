import torch
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
from models.tip_adapter.utils import build_cache_model
from models.xxx_clip import get_clip
from utils.show_image import save_image,show_image
from utils.validate import validate
from utils.tools import extract_from_batch_data
from dataset.build import build_dataloader
from utils.tools import pre_load_features
import clip
import json
from models.clip.clip import tokenize


def load_class_names(json_file):
    '''
    Returns: a list of class names
    '''
    with open(json_file, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return [v for k,v in data_dict.items()]

def raw_clip_test(cfg, logger, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    model, preprocess = clip.load('ViT-B/16', device)
    class_names = load_class_names(cfg.DATA.CLASS_NAMES)
    tokenized_cls_names = clip.tokenize(class_names).to(device)
    label_list = []
    logit_list = []
    for idx, batch_data in enumerate(tqdm(loader)):
        images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels: tensor shape=[bz]
        label_list.append(labels)
        images = images.reshape(-1, 3, cfg.DATA.INPUT_SIZE, cfg.DATA.INPUT_SIZE)
        with torch.no_grad():
            video_encode = model.encode_image(images)
            video_encode = video_encode / video_encode.norm(dim=-1, keepdim=True)
            video_encode = video_encode.reshape(batch_size, num_frames,
                                                -1)  # shape = [bz, num_frames, output_dim]
            video_encode = video_encode.mean(dim=1)
            text_features = model.encode_text(tokenized_cls_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = 100. * video_encode @ text_features.t()

        logit_list.append(logits)
    labels = torch.cat(label_list, dim=0).to(device)
    logits_ = torch.cat(logit_list, dim=0).to(device) # [56, 8]
    acc1, acc3, acc5, auc, f1 = validate(logits_, labels, plot=False, acc_only = False)
    logger.info('raw_clip test finished')
    logger.info(f"\nacc1: {acc1}\n"
                f"acc3: {acc3}\n"
                f"acc5: {acc5}\n"
                f"auc: {auc}\n"
            f"f1: {f1}\n")
    # for idx, batch_data in enumerate(tqdm(loader)):
    #     images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels: tensor shape=[bz]
    #     label_list.append(labels)
    #     images = images.reshape(batch_size, num_frames, 3, cfg.DATA.INPUT_SIZE, cfg.DATA.INPUT_SIZE).mean(dim=1) # [bz, c, h, w]
    #     with torch.no_grad():
    #         logits, _ = model(images, tokenized_cls_names)
    #     logit_list.append(logits)
    # labels = torch.cat(label_list, dim=0).to(device)
    # logits_ = torch.cat(logit_list, dim=0).to(device) # [56, 8]
    # acc1, acc3, acc5, auc, f1 = validate(logits_, labels, plot=False, acc_only = False)
    # logger.info('raw_clip test finished')
    # logger.info(f"\nacc1: {acc1}\n"
    #             f"acc3: {acc3}\n"
    #             f"acc5: {acc5}\n"
    #             f"auc: {auc}\n"
    #             f"f1: {f1}\n")
