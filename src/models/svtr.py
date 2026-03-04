"""SVTROCR with SVTR Backbone + CTC + Attention Decoder."""
import torch
import torch.nn as nn

from src.models.components import (
    TPSBlock,
    SVTRBackbone,
    TemporalTransformerFusion,
    PositionalEncoding,
    AttentionDecoder,
)

class SVTROCR(nn.Module):
    """
    Pipeline: [B, F, 3, H, W]
        → TPS-STN
        → SVTR Backbone     → [B*F, 192, 1, W']
        → TemporalFusion    → [B, 192, 1, W']
        → Transformer Enc   → [B, W', 192]   (encoder_out)
              ↓                      ↓
        CTC Head            Attention Decoder
        [B, W', C]          [B, max_len, C]
    """
    def __init__(
        self,
        num_classes: int,
        img_size: tuple  = (32, 128),
        num_fiducial: int = 20,
        transformer_heads: int  = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float   = 0.1,
        use_stn: bool    = True,
        max_len: int     = 25,       
        attn_weight: float = 0.5,    
    ):
        super().__init__()
        # SVTR Backbone outputs 192 channels by default (embed_dim[2])
        self.cnn_channels = 192 
        self.use_tps      = use_stn
        self.attn_weight  = attn_weight

        # 1. TPS-STN
        if self.use_tps:
            self.tps = TPSBlock(F=num_fiducial, I_size=img_size, I_channel_num=3)

        # 2. SVTR Backbone (Replacing ResNet)
        self.backbone = SVTRBackbone(
            img_size=img_size, 
            in_channels=3, 
            out_channels=self.cnn_channels
        )

        # 3. Temporal fusion
        self.fusion = TemporalTransformerFusion(channels=self.cnn_channels)

        # 4. Transformer Encoder 
        self.pos_encoder  = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels, nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 5a. CTC Head
        self.ctc_head = nn.Linear(self.cnn_channels, num_classes)

        # 5b. Attention Decoder
        self.attn_decoder = AttentionDecoder(
            num_classes  = num_classes,
            encoder_dim  = self.cnn_channels,
            hidden_dim   = 256,
            max_len      = max_len,
            num_heads    = 8,
            dropout      = dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor = None,
        target_lengths: torch.Tensor = None,
        return_sr: bool = False # Kept for compatibility with UniversalTrainer
    ) -> dict:
        
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)

        if self.use_tps:
            x_flat = self.tps(x_flat)

        features = self.backbone(x_flat)
        fused = self.fusion(features)

        seq = self.pos_encoder(fused.squeeze(2).permute(0, 2, 1))
        encoder_out = self.transformer(seq)    

        ctc_out = self.ctc_head(encoder_out).log_softmax(2)   

        # Standardize output as a dictionary
        result = {'ocr_logits': ctc_out}

        # Attention Decoder — chỉ khi training
        if self.training and targets is not None and target_lengths is not None:
            attn_out = self.attn_decoder(encoder_out, targets, target_lengths)  
            result['attn_logits'] = attn_out

        return result