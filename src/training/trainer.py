"""Universal Trainer class encapsulating the training, validation, and inference loop."""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence


class UniversalTrainer:
    """Encapsulates training, validation, inference, and Super-Resolution auxiliary logic."""
    def __init__(
        self,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)
        
        # Loss (OCR only)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
        
        if self.train_loader is not None:
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.LEARNING_RATE,
                steps_per_epoch=len(train_loader),
                epochs=config.EPOCHS
            )
            self.scaler = GradScaler()
        else:
            self.optimizer = None
            self.scheduler = None
            self.scaler = None
        
        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def _get_output_path(self, filename: str) -> str:
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        
        for batch in pbar:
            images, targets, target_lengths, labels_text, track_ids, hr_images = batch
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            #     # Model now returns a dict
            #     outputs = self.model(images, return_sr=True)
            #     ocr_logits = outputs['ocr_logits']
                
            #     # 1. OCR Loss
            #     preds_permuted = ocr_logits.permute(1, 0, 2)
            #     input_lengths = torch.full(
            #         size=(images.size(0),),
            #         fill_value=ocr_logits.size(1),
            #         dtype=torch.long,
            #         device=self.device
            #     )
            #     loss_ctc = self.criterion(preds_permuted.float(), targets, input_lengths, target_lengths)
            #     loss = loss_ctc
                
            #     postfix_dict = {'loss': f"{loss_ctc.item():.3f}"}
                
            #     # 2. Auxiliary SR Loss
            #     if 'sr_out' in outputs and hr_images.numel() > 0:
            #         hr_images = hr_images.to(self.device)
            #         # MSE against the clean HR target
            #         loss_sr = F.mse_loss(outputs['sr_out'].float(), hr_images.view(-1, 3, 32, 128).float())
            #         loss += 0.1 * loss_sr
            #         postfix_dict['sr'] = f"{loss_sr.item():.3f}"
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if self.config.MODEL_TYPE == "svtr":
                    # SVTR needs targets for Teacher Forcing in the Attention Decoder
                    outputs = self.model(images, targets=targets, target_lengths=target_lengths)
                else:
                    outputs = self.model(images, return_sr=True)
                
                # 1. Base OCR Loss (CTC)
                ocr_logits = outputs['ocr_logits']
                preds_permuted = ocr_logits.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=ocr_logits.size(1),
                    dtype=torch.long,
                    device=self.device
                )
                loss_ctc = self.criterion(preds_permuted.float(), targets, input_lengths, target_lengths)
                loss = loss_ctc
                
                postfix_dict = {'loss': f"{loss_ctc.item():.3f}"}
                    
                # 2. Auxiliary SR Loss (If active)
                if 'sr_out' in outputs and hr_images.numel() > 0:
                    hr_images = hr_images.to(self.device)
                    
                    sr_pred = F.interpolate(
                        outputs['sr_out'].float(), 
                        size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # MSE against the clean HR target
                    loss_sr = F.mse_loss(sr_pred, hr_images.view(-1, 3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH).float())
                    loss += 0.1 * loss_sr
                    postfix_dict['sr'] = f"{loss_sr.item():.3f}"

                # 3. Attention Decoder Loss (Cross Entropy)
                if 'attn_logits' in outputs:
                    attn_logits = outputs['attn_logits'] # [B, max_len, num_classes]
                    
                    # CTC targets are flat [sum_lengths]. We must pad them to [B, max_len] for CE Loss.
                    B = images.size(0)
                    max_len = attn_logits.size(1)
                    padded_targets = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
                    
                    start_idx = 0
                    for i, length in enumerate(target_lengths):
                        l = length.item()
                        padded_targets[i, :l] = targets[start_idx : start_idx + l]
                        start_idx += l
                    
                    # Calculate Cross Entropy (ignore_index=0 ensures padding/blanks aren't penalized)
                    # We flatten the batch and sequence dimensions to compute standard CE
                    loss_attn = F.cross_entropy(
                        attn_logits.reshape(-1, attn_logits.size(-1)), 
                        padded_targets.reshape(-1), 
                        ignore_index=0
                    )
                    
                    # Combine loss (weighting it by 0.5 to balance with CTC)
                    loss += 0.5 * loss_attn
                    postfix_dict['attn'] = f"{loss_attn.item():.3f}"

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'GRAD_CLIP', 2.0))
            
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_loss += loss.item()
            postfix_dict['lr'] = f"{self.scheduler.get_last_lr()[0]:.2e}"
            pbar.set_postfix(postfix_dict)
        
        return epoch_loss / len(self.train_loader)

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0}, []
        
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images, targets, target_lengths, labels_text, track_ids, _ = batch
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(images, return_sr=False)
                    ocr_logits = outputs['ocr_logits']
                    
                    input_lengths = torch.full((images.size(0),), ocr_logits.size(1), dtype=torch.long, device=self.device)
                    loss = self.criterion(ocr_logits.permute(1, 0, 2).float(), targets, input_lengths, target_lengths)
                    
                val_loss += loss.item()
                decoded_list = decode_with_confidence(ocr_logits, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    
                total_samples += len(labels_text)

        avg_val_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        return {'loss': avg_val_loss, 'acc': val_acc}, submission_data

    def save_submission(self, submission_data: List[str], filename: str = None) -> None:
        if filename is None:
            exp_name = self._get_exp_name()
            filename = f"submission_{exp_name}.txt"
        filepath = self._get_output_path(filename)
        with open(filepath, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filepath}")

    def save_model(self, path: str = None) -> None:
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        
        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            avg_train_loss = self.train_one_epoch()
            
            # If we have a validation loader, validate and save best
            if self.val_loader is not None:
                val_metrics, submission_data = self.validate()
                val_loss = val_metrics['loss']
                val_acc = val_metrics['acc']
                current_lr = self.scheduler.get_last_lr()[0]
                
                print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.2f}% | "
                      f"LR: {current_lr:.2e}")
                
                if val_acc > self.best_acc:
                    self.best_acc = val_acc
                    self.save_model()
                    print(f"  ⭐ Saved Best Model ({val_acc:.2f}%)")
                    if submission_data:
                        self.save_submission(submission_data)
            else:
                # 100% Data Mode (Submission Mode)
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.2e}")

        # Save final model if no validation was performed
        if self.val_loader is None:
            exp_name = self._get_exp_name()
            final_path = self._get_output_path(f"{exp_name}_final.pth")
            self.save_model(final_path)
            print(f"  💾 Saved final model to: {final_path}")
            
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        
        with torch.no_grad():
            for batch in loader:
                # Safely unpack 6 items
                images, _, _, _, track_ids, _ = batch
                images = images.to(self.device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(images, return_sr=False)
                    ocr_logits = outputs['ocr_logits']
                
                decoded_list = decode_with_confidence(ocr_logits, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        print(f"🔮 Running inference on test data...")
        
        results = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test Inference"):
                images, _, _, _, track_ids, _ = batch
                images = images.to(self.device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(images, return_sr=False)
                    ocr_logits = outputs['ocr_logits']
                    
                decoded_list = decode_with_confidence(ocr_logits, self.idx2char)
                
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        submission_data = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
        self.save_submission(submission_data, output_filename)