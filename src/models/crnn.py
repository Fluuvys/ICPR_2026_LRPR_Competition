"""Multi-frame CRNN architecture (Baseline) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import AttentionFusion, CNNBackbone, STNBlock


class MultiFrameCRNN(nn.Module):
    """
    Standard CRNN architecture adapted for Multi-frame input with optional STN alignment.
    Pipeline: Input (5 frames) -> [Optional STN] -> CNN Backbone -> Attention Fusion -> BiLSTM -> CTC Head
    """
    def __init__(self, num_classes: int, hidden_size: int = 256, rnn_dropout: float = 0.25, use_stn: bool = True):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        
        # 1. STN alignment (optional)
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)
        
        # 2. Feature Extractor (CNN Backbone)
        self.backbone = CNNBackbone(out_channels=self.cnn_channels)
        
        # 3. Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 4. Sequence Modeling (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=self.cnn_channels, # Height is collapsed to 1, so input is just channels
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout
        )
        
        # 5. Prediction Head
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]
        
        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat

        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        fused = self.fusion(features)    # [B, 512, 1, W']
        
        # --- Sequence Modeling ---
        # Prepare for RNN: [B, C, 1, W'] -> [B, W', C]
        # Squeeze height (1) and permute dimensions
        seq_input = fused.squeeze(2).permute(0, 2, 1)

        rnn_out, _ = self.rnn(seq_input) # [B, W', Hidden*2]
        out = self.head(rnn_out)         # [B, W', Num_Classes]
        
        return out.log_softmax(2)
    