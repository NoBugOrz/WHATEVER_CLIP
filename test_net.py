import torch
import torch.optim as optim
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm
from utils import show_image

from utils.show_image import save_image,show_image
from utils.validate import validate


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




def test(cfg, logger, test_loader, student_model):
    '''
    Testing the student model on the given dataset.
    '''
    logger.info('testing model on data from path:{}'.format(cfg.DATA.TRAIN_FILE))

    student_model.eval()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg.TRAIN.LR, eps=1e-4)

    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    logit_dic = {'model_logits':[]}
    label_list = []
    for idx, batch_data in enumerate(tqdm(test_loader)):
        images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels tensor shape=[bz]
        image_features, text_features, logits = student_model(images)

        label_list.append(labels)
        logit_dic['model_logits'].append(logits)


    labels = torch.cat(label_list)
    logit_dic['model_logits'] = torch.cat(logit_dic['model_logits'])

    acc1, acc3, acc5, auc, f1 = validate(logit_dic['model_logits'], labels, plot=False, acc_only = False)
    print(111)