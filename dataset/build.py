import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.teacher_detection.tld import TeacherDetection
import numpy as np
from PIL import Image
import cv2
import torch
from models.clip import clip
# from utils.tools import split_dataset
# import torch.distributed as dist
import os
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
from models.xxx_clip import load_class_names

class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,config,preprocess,device,ann_file,shot=0,type = 'train'):
        self.video_info = self.load_annotations(
            ann_file = ann_file,
            data_prefix = config.DATA.ROOT,
            num_frames = config.DATA.NUM_FRAMES,
            input_size = config.DATA.INPUT_SIZE,
            preprocess = preprocess,
            device = device,
            shot = shot,
            type = type,
            if_teacher = config.MODEL.IF_TEACHER,
            detector = TeacherDetection(config.MODEL.YOLO)
        )

    def prepare_frames(self, path, num_frames, if_teacher, detector, preprocess):
        if not os.path.exists(path):
            print(f"File {path} not found.")
            return None
        video_capture = cv2.VideoCapture(path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_ids = np.linspace(0, total_frames - 2, num_frames)
        frame_ids = np.floor(frame_ids).astype(int)
        for i in range(total_frames+1) :
            ret, frame = video_capture.read()
            if not ret:
                break
            if i in frame_ids:
                frames.append(frame)

        while len(frames) < num_frames:
            frames.extend(frames[:num_frames - len(frames)])
        video_capture.release()
        if if_teacher == 1:
            for i in range(len(frames)):
                frames[i] = detector(frames[i])
        frames = [
            preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))).unsqueeze(0) for c in
            frames]
        return frames

    @abstractmethod
    def load_annotations(self, ann_file,data_prefix,num_frames,input_size,preprocess,
                         device, shot, type, if_teacher, detector):
        """Load the annotation according to ann_file into video_infos."""

    def __len__(self):
        return len(self.video_info)

    def __getitem__(self, idx):
        return self.video_info[idx]


class VideoDataset(BaseDataset):
    def __init__(self, config,preprocess,device,ann_file,shot=0,type = 'train'):
        super().__init__(config,preprocess,device,ann_file,shot,type)
        self.labels_file = load_class_names(config.DATA.CLASS_NAMES)

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self, ann_file,data_prefix,num_frames,input_size,preprocess,
                         device, shot, type, if_teacher, detector):
        """Load annotation file to get video information."""
        video_infos = []
        class_counts = {}
        total_lines = sum(1 for line in open(ann_file, 'r'))
        if type == 'train':
            with open(ann_file, 'r') as fin:
                lines = fin.readlines()
                for idx in range(total_lines):  # Start from the last third
                    if idx % 5 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    if label in class_counts and class_counts[label] >= shot:
                        continue
                    data = self.prepare_frames(data_prefix + filename, num_frames, if_teacher, detector, preprocess)
                    if data is not None:
                        video_infos.append(dict(filename=filename, label=label, data=data))
                        if label not in class_counts:
                            class_counts[label] = 1
                        else:
                            class_counts[label] += 1
        elif type == 'test' or type == 'val':
            with open(ann_file, 'r') as fin:
                lines = fin.readlines()
                for idx in range(total_lines):  # Start from the last third
                    if idx % 5 == 0 and idx != 0:
                        progress = (idx / total_lines) * 100
                        print(f'Processed {idx} samples, progress: {progress:.2f}%')
                    line = lines[total_lines - idx - 1]
                    line_split = line.strip().split()
                    filename, label = line_split
                    label = int(label)
                    data = self.prepare_frames(data_prefix + filename, num_frames, if_teacher, detector, preprocess)
                    if data is not None:
                        video_infos.append(dict(filename=filename, label=label, data=data))

        return video_infos



class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices
    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    def __len__(self):
        return len(self.indices)
    def set_epoch(self, epoch):
        self.epoch = epoch

def build_dataloader(config, logger, loader_type:str):
    '''
    available loader_type: ['train', 'test', 'val', 'tip']
    '''
    assert loader_type in ['train', 'test', 'val', 'tip']

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    _, preprocess = clip.load(config.MODEL.ARCH, device=device)

    print('*'*10, f"building {loader_type} dataset", '*'*10)
    if loader_type == 'test':
        # ann_file = 'dataset/TBAD/test_files/all_names.txt'
        ann_file = os.path.join(config.DATA.TEST_FILE, "test_reordered_part{}.txt".format(4)) # 1-12,暂时用1
    elif loader_type == 'val':
        ann_file = os.path.join(config.DATA[loader_type.upper() + "_FILE"], '{}_{}shot.txt'.format(loader_type, 8)) # val use 8 shots
    elif loader_type == 'tip':
        ann_file = os.path.join(config.DATA[loader_type.upper() + "_FILE"],
                                '{}_{}shot.txt'.format(loader_type, 8))  # tip cached data use 8 shots
    else:
        ann_file = os.path.join(config.DATA[loader_type.upper() + "_FILE"], '{}_{}shot.txt'.format(loader_type, config.DATA.SHOTS))
    logger.info("Building {} dataset on data from path {}".format(loader_type, ann_file))
    data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=ann_file,
                        shot=config.DATA.SHOTS, type='train' if loader_type == 'train' else 'test')
    sampler = SubsetRandomSampler(np.arange(len(data)))
    loader = DataLoader(data, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler,
                        num_workers=12, pin_memory=True, drop_last=True)
    return data, loader

    '''tip adapter'''
    if is_tip:
        print('*'*10, "building tip adapter dataset", '*'*10)
        tip_ann_file = os.path.join(config.TIP_ADAPTER.DATA_FILE, 'tip_{}shot.txt'.format(config.DATA.SHOTS))
        tip_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=tip_ann_file,
                                  shot=config.DATA.SHOTS, type='train')
        sampler_tip = SubsetRandomSampler(np.arange(len(tip_data)))
        tip_loader = DataLoader(tip_data, batch_size=1, sampler=sampler_tip,
                                  num_workers=12, pin_memory=True, drop_last=True)
        return tip_data, tip_loader
    '''zero-shot'''
    if config.DATA.SHOTS == 0:
        test_ann_file = os.path.join(config.DATA.TEST_FILE, "test_reordered_part{}.txt".format(3))  # 1-12,暂时用1
        # test_ann_file = 'dataset/TBAD/test_files/all_names.txt'
        logger.info(f"testing on {test_ann_file}")
        test_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=test_ann_file, type='test')
        sampler_test = SubsetRandomSampler(np.arange(len(test_data)))
        test_loader = DataLoader(test_data, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler_test,
                                 num_workers=12, pin_memory=True, drop_last=True)
        return test_data, test_loader

    train_ann_file = os.path.join(config.DATA.TRAIN_FILE, "train_{}shot.txt".format(config.DATA.SHOTS))
    test_ann_file = os.path.join(config.DATA.TEST_FILE, "test_reordered_part{}.txt".format(1)) # 1-12,暂时用1
    val_ann_file = os.path.join(config.DATA.VAL_FILE, "val_set.txt")
    logger.info(f"training on {train_ann_file}")
    logger.info(f"testing on {test_ann_file}")
    logger.info(f"evaluating on {val_ann_file}")

    test_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=test_ann_file,type='test')
    sampler_test = SubsetRandomSampler(np.arange(len(test_data)))
    test_loader = DataLoader(test_data, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler_test,
                                 num_workers=12, pin_memory=True, drop_last=True)

    train_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=train_ann_file,
                                     shot=config.DATA.SHOTS, type='train')
    sampler_train = SubsetRandomSampler(np.arange(len(train_data)))
    train_loader = DataLoader(train_data, batch_size=config.TRAIN.BATCH_SIZE, sampler=sampler_train,
                             num_workers=12, pin_memory=True, drop_last=True)

    val_data = VideoDataset(config, preprocess=preprocess, device=device, ann_file=val_ann_file,
                              shot=config.DATA.SHOTS, type='val')
    sampler_val = SubsetRandomSampler(np.arange(len(val_data)))
    val_loader = DataLoader(val_data, batch_size=8, sampler=sampler_val,
                              num_workers=12, pin_memory=True, drop_last=True)

    return  train_data, test_data, val_data, train_loader , test_loader, val_loader

def split_dataset(dataset):
    # Step 1: Create a list of indices for each label
    label_to_indices = defaultdict(list)
    for idx, batch_data in enumerate(dataset):
        label = batch_data['label']
        label_to_indices[label].append(idx)

    # Step 2: Shuffle and split the indices for each label and add them to the new index lists
    indices1, indices2 = [], []
    for indices in label_to_indices.values():
        random.shuffle(indices)  # Shuffle the indices
        mid = len(indices) // 2
        if len(indices) % 2 == 1:  # Check if the number of samples is odd
            indices1.extend(indices[:mid+1])  # If odd, subset1 gets one more sample
            indices2.extend(indices[mid+1:])  # subset2 gets one less sample
        else:
            indices1.extend(indices[:mid])
            indices2.extend(indices[mid:])

    # Step 3: Create two Subset objects and two DataLoaders
    subset1 = Subset(dataset, indices1)
    subset2 = Subset(dataset, indices2)

    return subset1,subset2