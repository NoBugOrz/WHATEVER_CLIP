import torch
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
from utils import show_image

from utils.show_image import save_image,show_image

def extract_from_batch_data(batch_data,device):
    '''
    Returns:
        imgs : tensor of shape (bz * num_frames, C, H, W)
        labels : list of shape (bz * num_frames)
    '''
    images = batch_data['data']
    images = torch.stack(images)
    images = torch.transpose(images, 0, 1).to(device)  # [bz, num_frames, 1, c, h, w]
    images = images.squeeze(2).reshape(-1, 3, 224, 224)
    label_id = batch_data['label']  # list[] len=bz
    label_id = torch.tensor(label_id).to(device)
    return images, label_id




def train(cfg, logger, train_loader, student_model, teacher_model=None):
    '''
    Training the student model on the given dataset.
    '''
    logger.info('training model on data from path:{}'.format(cfg.DATA.TRAIN_FILE))

    if teacher_model is not None:
        logger.info('Use distillation in training')
        teacher_model.eval()

    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.TRAIN.EPOCHS * len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()

    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for k,v in student_model.named_parameters():
        if v.requires_grad:
            print(k)


    for cur_epoch in range(cfg.TRAIN.EPOCHS):
        loss_list = []
        for idx, batch_data in enumerate(tqdm(train_loader)):
            images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels tensor shape=[bz]
            image_features, text_features, logits = student_model(images)
            #save_image(images)
            show_image(images)
            preds = torch.argmax(logits, dim=1)


            loss = criterion(logits, labels)

            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        logger.info("In epoch:{}, loss:{}".format(cur_epoch, torch.tensor(loss_list).mean().item()))

