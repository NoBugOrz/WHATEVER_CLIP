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

def raw_clip_train(cfg, logger, train_loader, test_loader, val_loader, student_model, teacher_model=None):
    '''
    Training the student model on the given dataset.
    '''
    logger.info('training model on data from path:{}'.format(cfg.DATA.TRAIN_FILE))

    module_list = []

    teacher_model.train()

    logger.info('total trainable parameters:')
    for k, v in teacher_model.text_encoder.named_parameters():
        if '11' in k:
            v.requires_grad = True
        else:
            v.requires_grad = False
    for k, v in teacher_model.image_encoder.named_parameters():
        if '11' in k:
            v.requires_grad = True
        else:
            v.requires_grad = False
    for k,v in teacher_model.named_parameters():
        if v.requires_grad:
            print(k)

    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    print(f"优化器关联的参数组数量：{len(optimizer.param_groups)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for cur_epoch in range(cfg.TRAIN.EPOCHS):
        loss_list = []
        acc_dic = {'acc1':[], 'acc3':[], 'acc5':[]}
        for idx, batch_data in enumerate(train_loader):
            images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels tensor shape=[bz]
            image_features, text_features, logits = teacher_model(images)
            probs = logits.softmax(dim=-1)

            preds = torch.argmax(probs, dim=1)
            acc1, acc3, acc5 = validate(probs, labels,acc_only=True)
            acc_dic['acc1'].append(acc1)
            acc_dic['acc3'].append(acc3)
            acc_dic['acc5'].append(acc5)
            loss = criterion(logits, labels)

            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            #
            # for name, param in teacher_model.named_parameters():
            #     if param.grad is None:
            #         print(f"参数 {name} 未计算梯度（.grad 为 None）")
            #     else:
            #         # 检查梯度是否全为0（允许微小误差，避免浮点精度问题）
            #         is_zero = torch.allclose(param.grad, torch.zeros_like(param.grad), atol=1e-8)
            #         if is_zero:
            #             print(f"参数 {name} 的梯度全为0")
            #         else:
            #             # 可选：打印梯度的L2范数（判断梯度是否接近0）
            #             grad_norm = param.grad.norm().item()
            #             print(f"参数 {name} 的梯度非零，L2范数: {grad_norm:.6f}")

            optimizer.step()
            scheduler.step()

        logger.info(f"In epoch:{cur_epoch}, loss: {torch.tensor(loss_list).mean().item()}"
                    f" acc1: {np.array(acc_dic['acc1']).mean():.4f},"
                    f" acc3: {np.array(acc_dic['acc3']).mean():.4f},"
                    f" acc5: {np.array(acc_dic['acc5']).mean():.4f}")


