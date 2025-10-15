from models.clip.clip import load
from utils.logger import create_logger
from dataset.build import build_dataloader
from utils.config import get_config
import argparse
import os
import torch
from models.xxx_clip import xxx_clip,get_clip
from train_net import train, pre_load_features
from utils.raw_clip_test import raw_clip_train
from test_net import test

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/TBAD_L14.yaml')     
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--num_frames', type=int)
    parser.add_argument('--shots', type=int)
    parser.add_argument('--temporal_pooling', type=str)
    parser.add_argument('--test_file', type=str)
    parser.add_argument('--if_teacher', type=int)
    parser.add_argument('--output', type=str)
    parser.add_argument('--zs', type=int)
    parser.add_argument('--lp', type=int)
    parser.add_argument('--label_smooth', type=int)
    args = parser.parse_args()
    config = get_config(args)
    return args, config

def main(cfg, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if cfg.MODEL.USE_DISTILLATION:
        logger.info('loading raw_clip...')
        raw_clip = xxx_clip(cfg,device,is_teacher=True)
    else:
        raw_clip = None

    logger.info('building xxx_model...')
    student_model = xxx_clip(cfg,device,is_teacher=False)

    logger.info('loading dataloaders...')
    train_data, test_data, val_data, train_loader , test_loader, val_loader = build_dataloader(cfg, logger, is_tip=False)
    # raw_clip_train(cfg, logger, train_loader, test_loader, val_loader, student_model, teacher_model=raw_clip)
    train(cfg, logger, train_loader, test_loader, val_loader, student_model, teacher_model=raw_clip)
    test(cfg, logger, test_loader, student_model)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args, cfg = parse_option()
    logger = create_logger('logs')
    logger.info("Running with config:")
    logger.info(cfg)
    main(cfg, logger)