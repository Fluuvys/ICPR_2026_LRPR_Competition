import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor, 
    STNBlock, 
    SuperResolutionHead
)
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class NeuroMambaOCR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mamba_layers: int = 3,
        use_stn: bool = True,
        use_sr: bool = True
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.use_sr = use_sr
        
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        self.backbone = ResNetFeatureExtractor(pretrained=False)
        
        if self.use_sr:
            self.sr_head = SuperResolutionHead(in_channels=self.cnn_channels)

        if Mamba is None:
            raise ImportError("mamba_ssm is not installed.")
            
        # 🚨 FIX: The Pre-Norm "Shock Absorber"
        self.pre_norm = nn.LayerNorm(self.cnn_channels)
            
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=self.cnn_channels, d_state=16, d_conv=4, expand=2)
            for _ in range(mamba_layers)
        ])
        self.norm = nn.LayerNorm(self.cnn_channels)
        
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor, return_sr: bool = False) -> dict:
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)
        
        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
            
        features = self.backbone(x_aligned)
        
        features_view = features.view(b, f, self.cnn_channels, 1, -1)
        fused = features_view.mean(dim=1)
        
        seq_input = fused.squeeze(2).permute(0, 2, 1) # [B, W', C]
        
        # 🚨 FIX: Apply the Pre-Norm before entering the Mamba blocks
        seq_out = self.pre_norm(seq_input)
        
        for mamba_layer in self.mamba_blocks:
            # Mamba often benefits from residual connections inside the loop, 
            # but for sequence models this pre-norm usually prevents the NaN
            seq_out = mamba_layer(seq_out)
            
        seq_out = self.norm(seq_out)
        
        out = self.head(seq_out)
        ocr_logits = out.log_softmax(2)
        
        result = {'ocr_logits': ocr_logits}
        
        if self.use_sr and return_sr:
            result['sr_out'] = self.sr_head(features)
            
        return result