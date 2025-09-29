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

def train(cfg, logger, train_loader, student_model, teacher_model=None):
    '''
    Training the student model on the given dataset.
    '''
    logger.info('training model on data from path:{}'.format(cfg.DATA.TRAIN_FILE))

    if teacher_model is not None:
        logger.info('Use distillation in training')
        teacher_model = teacher_model.to(student_model.device) # make sure both models are on the same device
        teacher_model.eval()

    if cfg.TIP_ADAPTER.USE_TIP_ADAPTER == True:
        logger.info('Use tip adapter in training')
        tip_data, tip_loader = build_dataloader(cfg, logger, is_tip=True)
        build_cache_model(cfg=cfg,clip_model=teacher_model,train_loader_cache=tip_loader)

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

        logger.info(f"In epoch:{cur_epoch}, loss:{torch.tensor(loss_list).mean().item()}"
                    f" acc1:{np.array(acc_dic['acc1']).mean():.4f},"
                    f" acc3:{np.array(acc_dic['acc3']).mean():.4f},"
                    f" acc5:{np.array(acc_dic['acc5']).mean():.4f}")


