#!/usr/bin/env python3
"""Main entry point for unified OCR training pipeline."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR, ResTranOCRAlignMent
from src.models.mamba import NeuroMambaOCR
from src.models.svtr import SVTROCR
from src.models.new_svtr import svtrNew
from src.training.trainer import UniversalTrainer 
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-Frame OCR")
    parser.add_argument("-n", "--experiment-name", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, choices=["crnn", "restran", "mamba", "svtr", "new_svtr"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.0005, dest="learning_rate")
    parser.add_argument("--aug-level", type=str, choices=["full", "light"], default="full")
    parser.add_argument("--no-stn", action="store_true")
    # 🚨 NEW: Toggle for testing vs final run
    parser.add_argument("--full-dataset", action="store_true", help="Train on 100% of data (no val split)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    
    config.EXPERIMENT_NAME = args.experiment_name
    config.MODEL_TYPE = args.model
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.AUGMENTATION_LEVEL = args.aug_level
    if args.no_stn: config.USE_STN = False
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED)
    
    print(f"🚀 Configuration: {config.EXPERIMENT_NAME} | Model: {config.MODEL_TYPE}")
    
    if args.full_dataset:
        print("   MODE: 100% DATASET TRAINING (No Validation Split)")
    else:
        print("   MODE: STANDARD TRAINING (Train/Val Split Enabled)")

    # 1. Training Dataset
    train_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='train',
        full_train=args.full_dataset, 
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        seed=config.SEED,
        augmentation_level=config.AUGMENTATION_LEVEL,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    # 2. Validation Dataset (Only if not full_dataset)
    val_loader = None
    if not args.full_dataset:
        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            full_train=False,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            seed=config.SEED
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True
            )

    # 3. Initialize model
    if config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            use_sr=True 
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "mamba":
        model = NeuroMambaOCR(
            num_classes=config.NUM_CLASSES,
            mamba_layers=config.MAMBA_LAYERS,
            use_stn=config.USE_STN,
            use_sr=True 
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "svtr":
        model = SVTROCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            max_len=25 # Maximum license plate length
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "new_svtr":
        model = svtrNew(
            num_classes=config.NUM_CLASSES, 
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS, 
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT, 
            use_stn=config.USE_STN, 
            use_sr=True
        ).to(config.DEVICE)
    
    # 4. Initialize Universal Trainer
    trainer = UniversalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, 
        config=config,
        idx2char=config.IDX2CHAR
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()