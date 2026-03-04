"""Dataset for multi-frame license plate recognition with Super-Resolution targets."""
import os
import glob
import json
import random
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import get_train_transforms, get_val_transforms, get_light_transforms, get_degradation_transforms

class MultiFrameDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        mode: str = 'train',
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
    ):
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.augmentation_level = augmentation_level
        self.is_test = is_test
        self.full_train = full_train
        self.num_frames = 5  # Strictly enforced
        
        # STRICTLY CLEAN transform for HR target images
        self.clean_transform = get_val_transforms(img_height, img_width)
        
        if mode == 'train':
            self.transform = get_light_transforms(img_height, img_width) if augmentation_level == "light" else get_train_transforms(img_height, img_width)
            self.degrade = get_degradation_transforms()
        else:
            self.transform = get_val_transforms(img_height, img_width)
            self.degrade = None

        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))
        
        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        if is_test:
            self._index_test_samples(all_tracks)
        else:
            train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
            selected_tracks = train_tracks if mode == 'train' else val_tracks
            self._index_samples(selected_tracks)

        print(f"-> Total: {len(self.samples)} samples.")

    def _load_or_create_split(self, all_tracks: List[str], split_ratio: float) -> Tuple[List[str], List[str]]:
        if self.full_train: return all_tracks, []
        
        train_tracks, val_tracks = [], []
        if os.path.exists(self.val_split_file):
            try:
                with open(self.val_split_file, 'r') as f: val_ids = set(json.load(f))
            except Exception: val_ids = set()

            for t in all_tracks:
                if os.path.basename(t) in val_ids: val_tracks.append(t)
                else: train_tracks.append(t)
            
            if not val_tracks or (not any("Scenario-B" in t for t in val_tracks) and len(all_tracks) > 100):
                val_tracks = [] 

        if not val_tracks:
            scenario_b_tracks = [t for t in all_tracks if "Scenario-B" in t] or all_tracks
            val_size = max(1, int(len(scenario_b_tracks) * (1 - split_ratio)))
            random.Random(self.seed).shuffle(scenario_b_tracks)
            val_tracks = scenario_b_tracks[:val_size]
            
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]
            
            os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
            with open(self.val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)

        return train_tracks, val_tracks

    def _index_samples(self, tracks: List[str]) -> None:
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path): continue
            try:
                with open(json_path, 'r') as f: data = json.load(f)
                label = (data[0] if isinstance(data, list) else data).get('plate_text', data.get('license_plate', data.get('text', '')))
                if not label: continue
                
                track_id = os.path.basename(track_path)
                lr_files = sorted(glob.glob(os.path.join(track_path, "lr-*.png")) + glob.glob(os.path.join(track_path, "lr-*.jpg")))
                hr_files = sorted(glob.glob(os.path.join(track_path, "hr-*.png")) + glob.glob(os.path.join(track_path, "hr-*.jpg")))
                
                self.samples.append({'paths': lr_files, 'hr_paths': hr_files, 'label': label, 'is_synthetic': False, 'track_id': track_id})
                
                # Dual-Path Training: Train on degraded HR images too
                if self.mode == 'train' and hr_files:
                    self.samples.append({'paths': hr_files, 'hr_paths': hr_files, 'label': label, 'is_synthetic': True, 'track_id': track_id})
            except Exception: pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        for track_path in tqdm(tracks, desc="Indexing test"):
            lr_files = sorted(glob.glob(os.path.join(track_path, "lr-*.png")) + glob.glob(os.path.join(track_path, "lr-*.jpg")))
            if lr_files:
                self.samples.append({'paths': lr_files, 'hr_paths': [], 'label': '', 'is_synthetic': False, 'track_id': os.path.basename(track_path)})

    def __len__(self) -> int: return len(self.samples)

    def _load_and_pad_images(self, paths: List[str], is_target: bool, is_synthetic: bool) -> torch.Tensor:
        img_list = []
        max_h, max_w = 0, 0
        
        # First pass: Load and find the max dimensions in this specific sequence
        raw_imgs = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                max_h, max_w = max(max_h, h), max(max_w, w)
                raw_imgs.append(img)
                
        # Second pass: Pad all frames to the exact same high-resolution shape
        for img in raw_imgs:
            h, w, _ = img.shape
            if h != max_h or w != max_w:
                # Pad right and bottom to match the largest frame safely
                img = cv2.copyMakeBorder(img, 0, max_h - h, 0, max_w - w, cv2.BORDER_REPLICATE)
            img_list.append(img)

        # Pad sequence length to 5
        if len(img_list) == 0:
            dummy = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            img_list = [dummy] * self.num_frames
        else:
            while len(img_list) < self.num_frames:
                img_list.append(img_list[-1])
            img_list = img_list[:self.num_frames]

        transform_kwargs = {
            'image': img_list[0], 'image1': img_list[1],
            'image2': img_list[2], 'image3': img_list[3], 'image4': img_list[4]
        }

        if is_target:
            transformed = self.clean_transform(**transform_kwargs)
        else:
            # 1. Degrade at High Resolution first
            if is_synthetic and self.degrade:
                transform_kwargs = self.degrade(**transform_kwargs) 
            # 2. Apply Photometric shifts and Downscale to 32x128
            transformed = self.transform(**transform_kwargs)

        final_tensors = [
            transformed['image'], transformed['image1'],
            transformed['image2'], transformed['image3'], transformed['image4']
        ]
        
        return torch.stack(final_tensors, dim=0)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        
        # 1. Process Input Images (LR or Degraded HR)
        images_tensor = self._load_and_pad_images(item['paths'], is_target=False, is_synthetic=item['is_synthetic'])
        
        # 2. Process Target HR Images
        if not self.is_test and item.get('hr_paths'):
            hr_images_tensor = self._load_and_pad_images(item['hr_paths'], is_target=True, is_synthetic=False)
        else:
            hr_images_tensor = torch.empty(0) # Empty tensor for test/val
        
        # 3. Process Labels
        if self.is_test:
            target, target_len = [0], 1
        else:
            target = [self.char2idx[c] for c in item['label'] if c in self.char2idx]
            if len(target) == 0: target = [1]
            target_len = len(target)
            
        return images_tensor, torch.tensor(target, dtype=torch.long), target_len, item['label'], item['track_id'], hr_images_tensor

    @staticmethod
    def collate_fn(batch: List[Tuple]):
        images, targets, target_lengths, labels_text, track_ids, hr_images = zip(*batch)
        
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        if hr_images[0].numel() > 0:
            hr_images = torch.stack(hr_images, 0)
        else:
            hr_images = torch.empty(0)
            
        return images, targets, target_lengths, labels_text, track_ids, hr_images