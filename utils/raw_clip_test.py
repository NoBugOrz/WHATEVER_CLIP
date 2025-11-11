import numpy as np
import torch
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
from utils import show_image
from utils.show_image import save_image,show_image
from utils.validate import validate
from utils.tools import save_model
from utils.tools import extract_from_batch_data
from models.tip_adapter.utils import build_cache_model
from dataset.build import build_dataloader
import torch.nn as nn
import torch.nn.functional as F
from models.tip_adapter.utils import cls_acc,search_hp
from models.xxx_clip import get_clip
from test_net import test
from utils.scheduler import WarmupScheduler
from timm.loss import LabelSmoothingCrossEntropy


def raw_clip_train(cfg, logger, train_loader, test_loader, val_loader, student_model, teacher_model=None):
    '''
    Training the raw model on the given dataset.
    '''
    logger.info('training model on data from path:{}'.format(cfg.DATA.TRAIN_FILE))

    module_list = []
    # if student_model.spatial_temporal_module is not None:
    #     module_list.append('spatial_temporal_module')
    teacher_model.train()

    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = WarmupScheduler(optimizer=optimizer, warmup_epochs=int(cfg.TRAIN.EPOCHS * 0.3),
                                total_epochs=cfg.TRAIN.EPOCHS,
                                target_lr=cfg.TRAIN.LR, warmup_type='cosine', min_lr=cfg.TRAIN.LR * 0.02)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(train_loader))
    criterion = LabelSmoothingCrossEntropy()

    print(f"优化器关联的参数组数量：{len(optimizer.param_groups)}")

    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info('total trainable parameters:')
    for k, v in teacher_model.named_parameters():
        if '11' in k:
            v.requires_grad = True
        if k == 'clip.text_projection':
            v.requires_grad = True
        if k == 'clip.logit_scale':
            v.requires_grad = True
        if v.requires_grad:
            print(k)

    for cur_epoch in range(cfg.TRAIN.EPOCHS):
        teacher_model.train()
        loss_list = []
        acc_dic = {'acc1': [], 'acc3': [], 'acc5': []}
        for idx, batch_data in enumerate(train_loader):
            images, labels = extract_from_batch_data(batch_data,
                                                     device)  # images: tensor shape=[*, c, h, w],labels tensor shape=[bz]
            image_features, text_features, logits = teacher_model(images)
            probs = logits.softmax(dim=-1)
            acc1, acc3, acc5 = validate(probs, labels, acc_only=True)
            acc_dic['acc1'].append(acc1)
            acc_dic['acc3'].append(acc3)
            acc_dic['acc5'].append(acc5)
            loss = criterion(logits, labels)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        if cur_epoch % 5 == 0:
            logger.info(f"In epoch:{cur_epoch}, loss: {torch.tensor(loss_list).mean().item()}"
                        f" acc1: {np.array(acc_dic['acc1']).mean():.4f},"
                        f" acc3: {np.array(acc_dic['acc3']).mean():.4f},"
                        f" acc5: {np.array(acc_dic['acc5']).mean():.4f}")

        # eval
        if cur_epoch % 5 == 0:
            teacher_model.eval()
            val_acc_dic = {'acc1': [], 'acc3': [], 'acc5': []}
            for idx, batch_data in enumerate(test_loader):
                images, labels = extract_from_batch_data(batch_data, device)
                image_features, text_features, logits = teacher_model(images)
                probs = logits.softmax(dim=-1)
                val_acc1, val_acc3, val_acc5 = validate(probs, labels, acc_only=True)
                val_acc_dic['acc1'].append(val_acc1)
                val_acc_dic['acc3'].append(val_acc3)
                val_acc_dic['acc5'].append(val_acc5)

            logger.info(f"validation:  "
                        f" acc1: {np.array(val_acc_dic['acc1']).mean():.4f},"
                        f" acc3: {np.array(val_acc_dic['acc3']).mean():.4f},"
                        f" acc5: {np.array(val_acc_dic['acc5']).mean():.4f}")
            teacher_model.train()

    logger.info("now testing raw clip")
    test(cfg, logger, test_loader, teacher_model)
