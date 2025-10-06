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


def train_tip_adapter(cfg, logger, cache_keys, cache_values, student_model, dataloader):
    '''
    train and save tip_adapter model
    cache_keys: tensor shape=[512, 64]
    cache_values: tensor shape=[8, 8]
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(student_model.dtype).cuda() # in_dim=512, out_dim=64
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(dataloader))

    beta, alpha = cfg.TIP_ADAPTER.INIT_BETA, cfg.TIP_ADAPTER.INIT_ALPHA
    best_acc, best_epoch = 0.0, 0

    # make sure the student model is frozen, only adapter is trained
    student_model.eval()
    adapter.train()

    '''encoded text prompts'''
    clip_weights = student_model.text_encoder.short_cut # tensor, shape=[num_classes, 512]

    for train_idx in range(cfg.TIP_ADAPTER.TRAIN_EPOCH):
        adapter.train()

        correct_samples, all_samples = 0, 0
        loss_list = []
        acc_dic = {'acc1': [], 'acc3': [], 'acc5': []}
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg.TIP_ADAPTER.TRAIN_EPOCH))

        for i, batch_data in enumerate(tqdm(dataloader)):
            images, target = extract_from_batch_data(batch_data, device)  # images: tensor shape=[*, c, h, w],target tensor shape=[bz]
            images, target = images.cuda(), target.cuda()

            with torch.no_grad():
                image_features = student_model.image_encoder(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            probs = tip_logits.softmax(dim=-1)  # softmax to probability_like tensor
            acc1, acc3, acc5 = validate(probs, target, acc_only=True)
            acc_dic['acc1'].append(acc1)
            acc_dic['acc3'].append(acc3)
            acc_dic['acc5'].append(acc5)
            correct_samples += acc1 / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples,
                                                                       correct_samples, all_samples,
                                                                       sum(loss_list) / len(loss_list)))


def train(cfg, logger, train_loader, student_model, teacher_model=None):
    '''
    Training the student model on the given dataset.
    '''
    logger.info('training model on data from path:{}'.format(cfg.DATA.TRAIN_FILE))

    module_list = []

    if teacher_model is not None:
        logger.info('Use distillation in training')
        teacher_model = teacher_model.to(student_model.device) # make sure both models are on the same device
        teacher_model.eval()

    if cfg.TIP_ADAPTER.USE_TIP_ADAPTER == True:
        '''
        this part should be added when final logits are calculated
        '''
        logger.info('Use tip adapter in training')
        tip_data, tip_loader = build_dataloader(cfg, logger, is_tip=True)
        cache_keys, cache_values = build_cache_model(cfg=cfg,clip_model=teacher_model,train_loader_cache=tip_loader)

    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    logger.info('total trainable parameters:')
    for k,v in student_model.named_parameters():
        if v.requires_grad:
            print(k)


    for cur_epoch in range(cfg.TRAIN.EPOCHS):
        loss_list = []
        acc_dic = {'acc1':[], 'acc3':[], 'acc5':[]}
        for idx, batch_data in enumerate(train_loader):
            images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels tensor shape=[bz]
            image_features, text_features, logits = student_model(images)
            probs = logits.softmax(dim=-1)
            # save_image(images)
            # show_image(images)
            preds = torch.argmax(probs, dim=1)
            acc1, acc3, acc5 = validate(probs, labels,acc_only=True)
            acc_dic['acc1'].append(acc1)
            acc_dic['acc3'].append(acc3)
            acc_dic['acc5'].append(acc5)
            loss = criterion(logits, labels)

            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        logger.info(f"In epoch:{cur_epoch}, loss: {torch.tensor(loss_list).mean().item()}"
                    f" acc1: {np.array(acc_dic['acc1']).mean():.4f},"
                    f" acc3: {np.array(acc_dic['acc3']).mean():.4f},"
                    f" acc5: {np.array(acc_dic['acc5']).mean():.4f}")

    train_tip_adapter(cfg, logger, cache_keys, cache_values, student_model, train_loader)
