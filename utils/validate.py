import torch
from utils.tools import AverageMeter, classes, visual
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

@torch.no_grad()
def validate(output, label, plot = False):
    acc1_meter, acc5_meter,acc3_meter = AverageMeter(), AverageMeter(), AverageMeter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = label.clone().detach().to(device)
    all_preds = []
    all_labels = []
    all_probs = []
    for idx, similarity in enumerate(output):
        cur_label = label[idx]
        value1, indices_1 = similarity.topk(1, dim=-1)
        value3, indices_3 = similarity.topk(3, dim=-1)
        value5, indices_5 = similarity.topk(5, dim=-1)
        acc1, acc3 ,acc5 = 0, 0,0
        for i in range(1): # batch_size
            if indices_1[i] == cur_label:
                acc1 += 1
            if cur_label in indices_3:
                acc3 += 1
            if cur_label in indices_5:
                acc5 += 1
        acc1_meter.update(float(acc1) * 100,1)
        acc3_meter.update(float(acc3) * 100, 1)
        acc5_meter.update(float(acc5) * 100,1)
        all_preds.append(indices_1.cpu().numpy())
        all_labels.append(cur_label.cpu().numpy())
        probs = similarity.softmax(dim=-1).cpu().detach().numpy()
        if len(probs.shape) > 1:  # 如果probs有多个维度
            probs /= probs.sum(axis=1, keepdims=True)  # 归一化概率，使其和为1
        else:  # 如果probs只有一个维度
            probs /= probs.sum()  # 归一化概率，使其和为1
        if not np.isclose(probs.sum(), 1):
            probs = np.clip(probs, 0, 1)
            min_index = np.argmin(probs)
            sum = 0
            for i,num in enumerate(probs):
                if i != min_index:
                    sum += num
            probs[min_index] = 1 - sum
        all_probs.append(probs)
    # AUC and F1
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    f1 = f1_score(all_labels, all_preds, average='macro')
    if plot:
        cls = classes(config)
        labels = [sublist[1] for sublist in cls]

        cm = confusion_matrix(np.array(all_labels), np.array(all_preds))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Convert to percentages

        fig, ax = plt.subplots(figsize=(10, 10))  # Increase figure size
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax, shrink=0.7)  # Adjust the length of colorbar

        # Show all ticks
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'),  # Show 2 decimal places
                        ha='center', va='center', color='black')

        fig.tight_layout()  # Increase margin
        plt.savefig('confusion_matrix.png')

    return acc1_meter.avg, acc3_meter.avg, acc5_meter.avg, auc, f1