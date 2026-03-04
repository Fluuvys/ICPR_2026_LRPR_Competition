import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor, 
    PositionalEncoding, 
    STNBlock, 
    TemporalTransformerFusion, 
    SuperResolutionHead,
    LearnableAlignmentFusion,
    TemporalTransformerFusionNew
)

class ResTranOCR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
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
        
        self.fusion = TemporalTransformerFusionNew(channels=self.cnn_channels)
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
        
        # Primary OCR Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor, return_sr: bool = False) -> dict:
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w) 
        
        if self.use_stn:
            theta = self.stn(x_flat) # Define the matrix
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False) # Define the transformation
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False) # Apply the transformation
        else:
            x_aligned = x_flat
        
        features = self.backbone(x_aligned)  
        fused = self.fusion(features)       
        
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) 
        
        out = self.head(seq_out)             
        ocr_logits = out.log_softmax(2)
        
        result = {'ocr_logits': ocr_logits}
        
        if self.use_sr and return_sr:
            result['sr_out'] = self.sr_head(features)
            
        return result
    
