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

@torch.no_grad()
def test(cfg, logger, test_loader, student_model):
    '''
    Testing the student model on the given dataset.
    '''
    # logger.info('testing model on data from path:{}'.format(cfg.DATA.TEST_FILE))
    student_model.eval()
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_tip_adapter = cfg.TIP_ADAPTER.USE_TIP_ADAPTER
    if use_tip_adapter:
        val_data, val_loader = build_dataloader(cfg, logger, loader_type='val')
        cache_keys = torch.load(cfg.CACHE_DIR + '/keys_' + str(cfg.DATA.SHOTS) + "shots.pt")
        cache_values = torch.load(cfg.CACHE_DIR + '/values_' + str(cfg.DATA.SHOTS) + "shots.pt")
        if cfg.DATA.SHOTS != 0:
            '''加载训练好的adapter作为cache_keys的param'''
            pass
        perform_tip_adapter_test(cache_keys, cache_values, student_model, test_loader, val_loader)


    logit_dic = {'model_logits':[]}
    label_list = []
    for idx, batch_data in enumerate(tqdm(test_loader)):
        images, labels = extract_from_batch_data(batch_data,device) # images: tensor shape=[*, c, h, w],labels: tensor shape=[bz]
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
def perform_tip_adapter_test(cache_keys, cache_values, model, test_loader, val_loader):
    '''perform test when using tip adapter.'''
    clip_weights = student_model.text_encoder.short_cut.t()
    beta, alpha = cfg.TIP_ADAPTER.INIT_BETA, cfg.TIP_ADAPTER.INIT_ALPHA
    best_acc, best_epoch = 0.0, 0

    # Zero-shot CLIP
    clip_logits = 100. * val_features @ clip_weights
    acc = cls_acc(clip_logits, val_labels)
    print("\n**** Zero-shot CLIP's val accuracy: {:.2f}. ****\n".format(acc))

    for i, batch_data in enumerate(tqdm(train_loader)):
        images, target = extract_from_batch_data(batch_data, device)  # images: tensor shape=[*, c, h, w],target tensor shape=[bz]
