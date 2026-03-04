import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    SVTRBackbone,           # The only new variable
    PositionalEncoding, 
    STNBlock, 
    TemporalTransformerFusionNew, 
    SuperResolutionHead
)

class svtrNew(nn.Module):
    """
    ResTran pipeline with SVTR Backbone.
    Pipeline: [B, F, 3, H, W]
        -> STN
        -> SVTR Backbone (Outputs 192 channels)
        -> TemporalTransformerFusion
        -> Transformer Encoder
        -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        img_size: tuple = (32, 128),
        transformer_heads: int = 8,
        transformer_layers: int = 4, 
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        use_sr: bool = True
    ):
        super().__init__()
        self.cnn_channels = 256 
        self.use_stn = use_stn
        self.use_sr = use_sr
        
        # 1. Spatial Transformer Network (Control)
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. SVTR Backbone
        self.backbone = SVTRBackbone(
            img_size=img_size, 
            in_channels=3, 
            out_channels=self.cnn_channels
        )
        
        # 3. Super Resolution Head 
        if self.use_sr:
            self.sr_head = SuperResolutionHead(in_channels=self.cnn_channels)
        
        # 4. Temporal Fusion New 
        self.fusion = TemporalTransformerFusionNew(channels=self.cnn_channels)
        
        # 5. Transformer Sequence Encoder 
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 6. Primary CTC Head 
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
        
        # SVTR outputs: [B*F, 192, 1, W']
        features = self.backbone(x_aligned)  
        
        fused = self.fusion(features) # [B, 192, 1, W']      
        
        seq_input = fused.squeeze(2).permute(0, 2, 1) # [B, W', 192]
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) 
        
        out = self.head(seq_out)             
        ocr_logits = out.log_softmax(2)
        
        result = {'ocr_logits': ocr_logits}
        
        if self.use_sr and return_sr:
            # Note: SVTR output width might be slightly different than ResNet,
            # so the UniversalTrainer's F.interpolate() will handle the dynamic resizing.
            result['sr_out'] = self.sr_head(features)
            
        return result