from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
from utils.tools import extract_from_batch_data

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def build_cache_model(cfg, clip_model, train_loader_cache):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # bz = cfg.TRAIN.BATCH_SIZE
    num_frames = cfg.DATA.NUM_FRAMES
    output_dim = clip_model.output_dim # 也就是特征维度feature_dim
    if cfg.TIP_ADAPTER.LOAD_CACHE == False:
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg.TIP_ADAPTER.AUG_EPOCH):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg.TIP_ADAPTER.AUG_EPOCH))
                for i, batch_data  in enumerate(tqdm(train_loader_cache)):
                    images, target = extract_from_batch_data(batch_data,device)
                    images = images.cuda()
                    image_features = clip_model.encode_image(images) # [bz*num_frames, feature_dim]
                    '''
                    for each item in a batch, mean pooling features of num_frames to represent the whole item
                    [bz*num_frames, feature_dim] -> [bz, feature_dim]
                    *** use batchsize = 1 when building cache model or the dataloader would miss samples if len(samples) % batch_size != 0 ***
                    '''
                    image_features = image_features.reshape(-1, num_frames, output_dim)
                    image_features = image_features.mean(dim=1, keepdim=False)

                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))

        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg.CACHE_DIR + '/keys_' + str(cfg.DATA.SHOTS) + "shots.pt")
        torch.save(cache_values, cfg.CACHE_DIR + '/values_' + str(cfg.DATA.SHOTS) + "shots.pt")

    else:
        cache_keys = torch.load(cfg.CACHE_DIR + '/keys_' + str(cfg.DATA.SHOTS) + "shots.pt")
        cache_values = torch.load(cfg.CACHE_DIR + '/values_' + str(cfg.DATA.SHOTS) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader):
    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")

    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")

    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None):
    if cfg.TIP_ADAPTER.SEARCH_HP == True:

        beta_list = [i * (cfg.TIP_ADAPTER.SEARCH_SCALE[0] - 0.1) / cfg.TIP_ADAPTER.SEARCH_STEP[0] + 0.1 for i in
                     range(cfg.TIP_ADAPTER.SEARCH_STEP[0])]
        alpha_list = [i * (cfg.TIP_ADAPTER.SEARCH_SCALE[1] - 0.1) / cfg.TIP_ADAPTER.SEARCH_STEP[1] + 0.1 for i in
                      range(cfg.TIP_ADAPTER.SEARCH_STEP[1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                if adapter:
                    affinity = adapter(features)
                else:
                    affinity = features @ cache_keys

                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                clip_logits = 100. * features @ clip_weights
                tip_logits = clip_logits + cache_logits * alpha
                acc = cls_acc(tip_logits, labels)

                if acc > best_acc:
                    print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha