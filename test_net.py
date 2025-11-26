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
from models.tip_adapter.utils import cls_acc, search_hp
import torch.nn as nn
import os
'''
after multiple testing, in zero-shot, the best alpha and beta is [1.59, 4.75], yield acc1: 48, acc3:71, acc5:80
'''

path_dic = {'ViT-B/16':"shots.pt",
            'ViT-L/14':"shots_L14.pt",
            'ViT-L/14@336px':"shots_L14@336px.pt"
            }

@torch.no_grad()
def test(cfg, logger, test_loader, student_model):
    '''
    Testing the student model on the given dataset.
    '''
    # logger.info('testing model on data from path:{}'.format(cfg.DATA.TEST_FILE))
    student_model.eval()
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    if cfg.TRAIN.NUM_GPUS > 1:
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cuda")

    use_tip_adapter = cfg.TIP_ADAPTER.USE_TIP_ADAPTER
    if use_tip_adapter:
        post_path = path_dic[cfg.MODEL.ARCH]
        cache_keys = torch.load(cfg.CACHE_DIR + '/keys_' + str(8) + post_path)
        cache_values = torch.load(cfg.CACHE_DIR + '/values_' + str(8) + post_path)
        if cfg.DATA.SHOTS != 0:
            '''加载训练好的adapter作为cache_keys的param'''
            pass
        else:
            run_tip_adapter(cfg, logger, cache_keys, cache_values, student_model, test_loader, device)


    logit_dic = {'model_logits':[]}
    label_list = []
    for idx, batch_data in enumerate(tqdm(test_loader)):
        images, labels = extract_from_batch_data(batch_data,device,cfg) # images: tensor shape=[*, c, h, w],labels: tensor shape=[bz]
        # save_image(images, 'images/')
        image_features, text_features, logits = student_model(images)

        label_list.append(labels)
        logit_dic['model_logits'].append(logits)

    labels = torch.cat(label_list)
    logit_dic['model_logits'] = torch.cat(logit_dic['model_logits'])

    acc1, acc3, acc5, auc, f1 = validate(logit_dic['model_logits'], labels, plot=False, acc_only = False)
    logger.info('test finished')
    logger.info(f"\nacc1: {acc1}\n"
                f"acc3: {acc3}\n"
                f"acc5: {acc5}\n"
                f"auc: {auc}\n"
                f"f1: {f1}\n")

@torch.no_grad()
def run_tip_adapter(cfg, logger, cache_keys, cache_values, model, test_loader, device):
    '''run tip adapter. non-trainable adapter'''
    print("*"*10, "running tip adapter", "*"*10)
    cache_values = cache_values.to(torch.float32)
    cache_keys = cache_keys.to(torch.float32)
    clip_weights = model.text_encoder.short_cut.t().to(torch.float32)
    val_data, val_loader = build_dataloader(cfg, logger, loader_type='val')
    val_features, val_labels, val_logits = pre_load_features(model, val_loader, cfg)
    val_features = val_features.to(torch.float32)
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).cuda()  # in_dim=512, out_dim=8
    adapter.weight = nn.Parameter(cache_keys.t())
    adapter = adapter.eval()
    # clip_logits = 100. * val_features @ clip_weights
    clip_logits = val_logits
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    beta, alpha = cfg.TIP_ADAPTER.INIT_BETA, cfg.TIP_ADAPTER.INIT_ALPHA
    affinity = adapter(val_features)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, val_labels)
    print("**** Tip-Adapter's val accuracy: {:.2f}. ****\n".format(acc))
    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights, adapter=adapter, device=device)

    print("\n-------- Evaluating on the test set. --------")

    # Zero-shot CLIP
    test_features, test_labels, test_logits = pre_load_features(model, test_loader, cfg)
    test_features = test_features.to(torch.float32)
    # clip_logits = 100. * test_features @ clip_weights
    clip_logits = test_logits
    acc = cls_acc(clip_logits, test_labels)
    logger.info("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha

    acc1, acc3, acc5, auc, f1 = validate(tip_logits, test_labels, plot=False, acc_only=False)
    logger.info("Tip-Adapter's test accuracy:")
    logger.info(f"\nacc1: {acc1}\n"
                f"acc3: {acc3}\n"
                f"acc5: {acc5}\n"
                f"auc: {auc}\n"
                f"f1: {f1}\n")