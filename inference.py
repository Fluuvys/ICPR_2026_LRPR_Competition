#!/usr/bin/env python3
"""Standalone inference script for generating Codabench submissions."""
import argparse
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranOCR
from src.models.mamba import NeuroMambaOCR
from src.models.svtr import SVTROCR
from src.models.new_svtr import svtrNew  
from src.training.trainer import UniversalTrainer 

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference on Blind Test Set")
    # 🚨 ADDED: "new_svtr" to choices
    parser.add_argument("-m", "--model", type=str, choices=["restran", "mamba", "svtr", "new_svtr"], required=True, help="Model architecture used")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to the trained .pth file")
    parser.add_argument("-o", "--output", type=str, default="submission.txt", help="Output filename for Codabench")
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config()
    config.MODEL_TYPE = args.model
    
    print(f"🚀 INFERENCE MODE | Model: {config.MODEL_TYPE}")
    print(f"📦 Loading weights from: {args.weights}")
    
    # 1. Initialize the correct model architecture
    if config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            use_sr=False # Turn off SR for raw inference speed
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "mamba":
        model = NeuroMambaOCR(
            num_classes=config.NUM_CLASSES,
            mamba_layers=config.MAMBA_LAYERS,
            use_stn=config.USE_STN,
            use_sr=False 
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "svtr":
        model = SVTROCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            max_len=25 
        ).to(config.DEVICE)
    elif config.MODEL_TYPE == "new_svtr":
        model = svtrNew(
            num_classes=config.NUM_CLASSES,
            img_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=4,     # Matches our bumped capacity
            transformer_ff_dim=2048,  # Matches our bumped capacity
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
            use_sr=False              # SR not needed for inference
        ).to(config.DEVICE)

    # 2. Load the trained weights
    if not os.path.exists(args.weights):
        print(f"❌ ERROR: Weight file not found at {args.weights}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(args.weights, map_location=config.DEVICE), strict=False)
    model.eval()

    # 3. Load the Blind Test Dataset
    if not os.path.exists(config.TEST_DATA_ROOT):
        print(f"❌ ERROR: Test data not found at {config.TEST_DATA_ROOT}")
        sys.exit(1)
        
    print(f"📂 Scanning Test Data: {config.TEST_DATA_ROOT}")
    test_ds = MultiFrameDataset(
        root_dir=config.TEST_DATA_ROOT,
        mode='val',
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        is_test=True 
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn, num_workers=config.NUM_WORKERS
    )

    # 4. Use the UniversalTrainer's built-in inference logic
    # We pass None for train/val loaders since we aren't training
    trainer = UniversalTrainer(
        model=model,
        train_loader=None,
        val_loader=None,
        config=config,
        idx2char=config.IDX2CHAR
    )
    
    # 5. Generate and save predictions
    trainer.predict_test(test_loader, output_filename=args.output)

if __name__ == "__main__":
    main()