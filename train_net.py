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

# torch.autograd.set_detect_anomaly(True)

def pre_load_features(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, labels, logits = [], [], []
    for i, batch_data in enumerate(tqdm(loader)):
        images, target = extract_from_batch_data(batch_data, device)  # images: tensor shape=[*, c, h, w],target tensor shape=[bz]
        images, target = images.cuda(), target.cuda()

        with torch.no_grad():
            image_features, text_features, clip_logits = model(images)

        features.append(image_features)
        labels.append(target)
        logits.append(clip_logits)

    return torch.cat(features), torch.cat(labels), torch.cat(logits)

def train_tip_adapter(cfg, logger, cache_keys, cache_values, student_model, train_loader,
                      val_features, val_labels, test_features, test_labels, clip_weights, test_logits):
    '''
    train and save tip_adapter model
    cache_keys: tensor shape=[512, 8]
    cache_values: tensor shape=[8, 8]
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(student_model.dtype).cuda() # in_dim=512, out_dim=8
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.Adam(adapter.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    beta, alpha = cfg.TIP_ADAPTER.INIT_BETA, cfg.TIP_ADAPTER.INIT_ALPHA
    best_acc, best_epoch = 0.0, 0

    # make sure the student model is frozen, only adapter is trained
    student_model.eval()
    adapter.train()


    for train_idx in range(cfg.TIP_ADAPTER.TRAIN_EPOCH):
        adapter.train()

        correct_samples, all_samples = 0, 0
        loss_list = []
        acc_dic = {'acc1': [], 'acc3': [], 'acc5': []}
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg.TIP_ADAPTER.TRAIN_EPOCH))

        for i, batch_data in enumerate(tqdm(train_loader)):
            images, target = extract_from_batch_data(batch_data, device)  # images: tensor shape=[*, c, h, w],target tensor shape=[bz]
            images, target = images.cuda(), target.cuda()

            with torch.no_grad():
                image_features, text_features, clip_logits = student_model(images)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            tip_logits = clip_logits + cache_logits * alpha

            loss = criterion(tip_logits, target)

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
        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        # clip_logits = (100. * test_features @ clip_weights).softmax(dim=-1)

        clip_logits = test_logits

        tip_logits = clip_logits + cache_logits * alpha

        acc = cls_acc(tip_logits, test_labels)

        print("**** tip_adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg.CACHE_DIR + "/best_F_" + str(cfg.DATA.SHOTS) + "shots.pt")

    adapter.weight = torch.load(cfg.CACHE_DIR + "/best_F_" + str(cfg.DATA.SHOTS) + "shots.pt")
    print(f"**** After fine-tuning, tip_adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    print("\n-------- Searching hyperparameters on the val set. --------")

    # Search Hyperparameters
    best_beta, best_alpha = search_hp(cfg, cache_keys, cache_values, val_features, val_labels, clip_weights,
                                      adapter=adapter)

    print("\n-------- Evaluating on the test set. --------")

    affinity = adapter(test_features)
    cache_logits = ((-1) * (best_beta - best_beta * affinity)).exp() @ cache_values

    tip_logits = clip_logits + cache_logits * best_alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** tip_adapter-F's test accuracy: {:.2f}. ****\n".format(max(best_acc, acc)))



def train(cfg, logger, train_loader, test_loader, val_loader, student_model, teacher_model=None):
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
        raw_clip_model = get_clip(cfg, is_teacher=True)
        cache_keys, cache_values = build_cache_model(cfg=cfg,clip_model=raw_clip_model,train_loader_cache=tip_loader)

    student_model.train()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    print(f"优化器关联的参数组数量：{len(optimizer.param_groups)}")

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

            # for name, param in student_model.named_parameters():
            #     if 'clip' in name:
            #         continue
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

    if cfg.TIP_ADAPTER.USE_TIP_ADAPTER == True:
        val_features, val_labels, val_logits = pre_load_features(student_model, val_loader) # [num_samples, 512]
        test_features, test_labels, test_logits = pre_load_features(student_model, test_loader) # [num_samples, 512]
        clip_weights = student_model.text_encoder.short_cut.t()
        # clip_weights = student_model.text_encoder._forward(student_model.text_encoder._tokens)
        train_tip_adapter(cfg, logger, cache_keys, cache_values, student_model, train_loader, val_features, val_labels, test_features, test_labels, clip_weights, test_logits) # [num_samples]
