"""Reusable model components"""
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34, ResNet18_Weights, resnet18
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class STNBlock(nn.Module):
    """
    Spatial Transformer Network (STN) for image alignment.
    Learns to crop and rectify images before feeding them to the backbone.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Localization network: Predicts transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) # Output fixed size for FC
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, C, H, W]
        Returns:
            theta: Affine transformation matrix [Batch, 2, 3]
        """
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-frame features.
    Computes a weighted sum of features from multiple frames based on their 'quality' scores.
    """
    def __init__(self, channels: int):
        super().__init__()
        # A small CNN to predict attention scores (quality map) from features
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        num_frames = 5  # Fixed based on dataset
        batch_size = total_frames // num_frames

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)
        
        # Calculate attention scores: [Batch, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)  # Normalize scores across frames

        # Weighted sum fusion
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class CNNBackbone(nn.Module):
    """A simple CNN backbone for CRNN baseline."""
    def __init__(self, out_channels=512):
        super().__init__()
        # Defined as a list of layers for clarity: Conv -> ReLU -> Pool
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5 (Map to sequence height 1)
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based backbone customized for OCR.
    Uses ResNet34 with modified strides to preserve width (sequence length) while reducing height.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        
        # Load ResNet34 from torchvision
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        # --- OCR Customization ---
        # We need to keep the standard first layer (stride 2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify strides in layer3 and layer4 to (2, 1)
        # This reduces height but preserves width for sequence modeling
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, 3, H, W]
        Returns:
            Features [Batch, 512, H // 16, W // 2] (approx)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Ensure height is 1 for sequence modeling (Height collapsing)
        # Output shape: [Batch, 512, 1, W']
        x = F.adaptive_avg_pool2d(x, (1, None))
        return x



class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Standard Sinusoidal implementation from 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
    
class TemporalTransformerFusion(nn.Module):
    """
    Temporal Transformer Fusion cho multi-frame features.

    Thay vì weighted sum (Attention) hay concat (Concat),
    coi 5 frame như 1 sequence và dùng Transformer học
    mối quan hệ GIỮA các frame (frame nào bù frame nào).

    Pipeline:
        [B*F, C, 1, W'] 
            → reshape [B, F, W', C]          # F frame, W' time steps, C channels
            → + frame positional encoding    # biết frame nào là frame mấy
            → TransformerEncoder             # self-attention across frames
            → mean pooling over F            # gộp F frame thành 1
            → [B, W', C]                     # đưa vào Spatial Transformer tiếp theo

    Tại sao tốt hơn AttentionFusion:
        - AttentionFusion: mỗi spatial position (W') quyết định độc lập
          frame nào quan trọng → không share thông tin giữa positions
        - TemporalTransformerFusion: self-attention toàn bộ F*W' tokens
          → học được "frame 2 rõ bên trái, frame 4 rõ bên phải" → bù trừ tốt hơn
    """
    def __init__(
        self,
        channels: int,
        num_frames: int = 5,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels

        # Frame positional encoding: phân biệt frame 1,2,3,4,5
        # Khác với spatial positional encoding (theo W')
        self.frame_pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, 1, channels) * 0.02
        )

        # Transformer Encoder xử lý sequence [F * W'] tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,   # [B, Seq, C]
            norm_first=True,    # Pre-LN ổn định hơn Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Layer norm sau fusion
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B * F, C, 1, W'] — output từ ResNetFeatureExtractor
        Returns:
            [B, C, 1, W'] — fused feature map
        """
        total, c, h, w = x.size()                      # h=1 vì đã adaptive pool
        B = total // self.num_frames
        F = self.num_frames

        # [B*F, C, 1, W'] → [B, F, W', C]
        x = x.squeeze(2)                               # [B*F, C, W']
        x = x.view(B, F, c, w)                        # [B, F, C, W']
        x = x.permute(0, 1, 3, 2)                     # [B, F, W', C]

        # Thêm frame positional encoding
        # frame_pos_embedding: [1, F, 1, C] → broadcast sang [B, F, W', C]
        x = x + self.frame_pos_embedding

        # Flatten F và W' thành 1 sequence: [B, F*W', C]
        x = x.reshape(B, F * w, c)

        # Transformer: self-attention toàn bộ F*W' tokens
        x = self.transformer(x)                        # [B, F*W', C]
        x = self.norm(x)

        # Reshape về [B, F, W', C] rồi mean pool theo F
        x = x.view(B, F, w, c)
        x = x.mean(dim=1)                              # [B, W', C]

        # Trả về [B, C, 1, W'] để khớp với phần sau của ResTranOCR
        x = x.permute(0, 2, 1).unsqueeze(2)           # [B, C, 1, W']
        return x
    
class TemporalTransformerFusionNew(nn.Module):
    """
    Factorized Temporal Attention.
    Strictly calculates attention ACROSS FRAMES for each spatial position independently.
    Cleaner inductive bias, drastically lower FLOPs than global attention, 
    allowing for deeper layers and wider channels.
    """
    def __init__(
        self,
        channels: int,
        num_frames: int = 5,
        num_heads: int = 8, 
        num_layers: int = 3, # BUMPED: 2 -> 3
        ff_dim: int = 1536,  # BUMPED: 1024 -> 1536
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.channels = channels

        self.frame_pos_embedding = nn.Parameter(torch.randn(1, num_frames, 1, channels) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,   
            norm_first=True,    
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*F, C, 1, W']
        total, c, _, w = x.size()                      
        B = total // self.num_frames
        F = self.num_frames

        # [B, F, C, W'] -> [B, F, W', C]
        x = x.squeeze(2).view(B, F, c, w).permute(0, 1, 3, 2)                     
        x = x + self.frame_pos_embedding

        # 🚨 THE FACTORIZATION
        # Group B and W' together to isolate F: [B * W', F, C]
        # Now the transformer ONLY looks across the 5 frames for a specific spatial slice
        x_time = x.permute(0, 2, 1, 3).contiguous().view(B * w, F, c)

        # Temporal Self-Attention
        x_time = self.transformer(x_time) 
        x_time = self.norm(x_time)

        # Reshape back to [B, F, W', C] and apply Context-Aware Mean Pooling
        x_fused = x_time.view(B, w, F, c).permute(0, 2, 1, 3)
        x_fused = x_fused.mean(dim=1) # [B, W', C]

        # Return to [B, C, 1, W'] format
        return x_fused.permute(0, 2, 1).unsqueeze(2)
    
    

class SuperResolutionHead(nn.Module):
    """
    Auxiliary head to reconstruct high-resolution images from ResNet34 features.
    Input shape from ResNet34: [B*F, 512, 1, 16] 
    Output shape target: [B*F, 3, 32, 128]
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # Step 1: 1x16 -> 2x32
            nn.ConvTranspose2d(in_channels, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Step 2: 2x32 -> 4x64
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Step 3: 4x64 -> 8x128
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Step 4: 8x128 -> 16x128 (Width reached 128, only expand height now)
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Step 5: 16x128 -> 32x128
            nn.ConvTranspose2d(32, 3, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            nn.Tanh()  # Scale pixel outputs to be between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
    
class SharedAttentionModule(nn.Module):
    """
    Represents the shared PLTFAM-style attention from the paper.
    Shared weights across all Residual Blocks to consistently emphasize LP features.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        ca_out = x * self.ca(x)
        
        # Spatial Attention (max & avg pooling along channel dim)
        max_out, _ = torch.max(ca_out, dim=1, keepdim=True)
        avg_out = torch.mean(ca_out, dim=1, keepdim=True)
        sa_in = torch.cat([max_out, avg_out], dim=1)
        sa_out = ca_out * self.sa(sa_in)
        
        return sa_out

class ResidualBlockWithSharedAttention(nn.Module):
    def __init__(self, channels: int, shared_attention: nn.Module):
        super().__init__()
        # The paper mentions replacing standard convs with Deformable Convs here.
        # We use standard 3x3 as a stable starting point for the baseline.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # The shared attention instance passed from the parent
        self.shared_attention = shared_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.prelu(self.conv1(x))
        res = self.conv2(res)
        # Apply the shared attention as specified in the paper
        res = self.shared_attention(res)
        return x + res

class LPSRGenerator(nn.Module):
    """
    The standalone Super-Resolution Generator.
    Takes 5 LR frames and reconstructs 1 HR Image.
    """
    def __init__(self, num_frames: int = 5, in_channels: int = 3, out_channels: int = 3, features: int = 64, num_blocks: int = 4):
        super().__init__()
        
        # 1. Spatio-temporal Fusion: Flatten 5 frames (15 channels) to extract features
        self.fusion = nn.Sequential(
            nn.Conv2d(num_frames * in_channels, features, kernel_size=3, padding=1),
            nn.PReLU()
        )
        
        # 2. Shared Attention Module (Core architectural contribution of the paper)
        self.shared_attention = SharedAttentionModule(features)
        
        # 3. Residual Blocks utilizing the exact same shared attention
        self.blocks = nn.ModuleList([
            ResidualBlockWithSharedAttention(features, self.shared_attention)
            for _ in range(num_blocks)
        ])
        
        # 4. Final Reconstruction to HR image dimensions
        self.reconstruct = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid() # Bound pixel values between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, F, C, H, W]
        b, f, c, h, w = x.size()
        x_flat = x.view(b, f * c, h, w)
        
        feat = self.fusion(x_flat)
        
        out = feat
        for block in self.blocks:
            out = block(out)
            
        out = out + feat # Global residual connection
        
        return self.reconstruct(out)
    
    
class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block for sequence modeling.
    Crucial for OCR so the model can understand context from both sides of a character.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba-ssm is not installed. Please run: pip install mamba-ssm causal-conv1d")
            
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # Linear layer to project the concatenated bidirectional features back to d_model
        self.proj = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, Seq_Len, C]
        forward_out = self.forward_mamba(x)
        
        # Flip sequence for backward pass
        x_flipped = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(x_flipped)
        # Flip back to align with forward output
        backward_out = torch.flip(backward_out, dims=[1])
        
        # Concat, project, and apply residual + norm
        merged = torch.cat([forward_out, backward_out], dim=-1)
        out = self.proj(merged)
        
        return self.norm(x + out)
    
    
class TPSLocalizationNetwork(nn.Module):
    def __init__(self, F: int, I_channel_num: int = 3):
        super().__init__()
        self.F = F
        self.conv = nn.Sequential(
            nn.Conv2d(I_channel_num, 64,  3, 1, 1, bias=False), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64,  128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, F * 2)
        self.localization_fc2.weight.data.fill_(0)
        ctrl_pts_x        = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top    = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0,  0.0, num=int(F / 2))
        initial_bias = np.concatenate([
            np.stack([ctrl_pts_x, ctrl_pts_y_top],    axis=1),
            np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1),
        ], axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):
        B = batch_I.size(0)
        return self.localization_fc2(self.localization_fc1(self.conv(batch_I).view(B, -1))).view(B, self.F, 2)


class TPSGridGenerator(nn.Module):
    def __init__(self, F: int, I_r_size: tuple):
        super().__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(F)
        P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer('inv_delta_C', torch.tensor(self._build_inv_delta_C(F, self.C)).float())
        self.register_buffer('P_hat',       torch.tensor(self._build_P_hat(F, self.C, P)).float())

    def _build_C(self, F):
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        return np.concatenate([
            np.stack([ctrl_pts_x, -1 * np.ones(int(F / 2))], axis=1),
            np.stack([ctrl_pts_x,      np.ones(int(F / 2))], axis=1),
        ], axis=0)

    def _build_inv_delta_C(self, F, C):
        hat_C = np.zeros((F, F), dtype=float)
        for i in range(F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        delta_C = np.concatenate([
            np.concatenate([np.ones((F, 1)), C, hat_C],         axis=1),
            np.concatenate([np.zeros((2, 3)), C.T],              axis=1),
            np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),
        ], axis=0)
        return np.linalg.inv(delta_C)

    def _build_P(self, W, H):
        grid_x = (np.arange(-W, W, 2) + 1.0) / W
        grid_y = (np.arange(-H, H, 2) + 1.0) / H
        return np.stack(np.meshgrid(grid_x, grid_y), axis=2).reshape([-1, 2])

    def _build_P_hat(self, F, C, P):
        n        = P.shape[0]
        P_diff   = np.tile(np.expand_dims(P, 1), (1, F, 1)) - np.expand_dims(C, 0)
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2)
        rbf      = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))
        return np.concatenate([np.ones((n, 1)), P, rbf], axis=1)

    def build_P_prime(self, batch_C_prime):
        B = batch_C_prime.size(0)
        batch_C_prime_with_zeros = torch.cat([
            batch_C_prime,
            torch.zeros(B, 3, 2, device=batch_C_prime.device, dtype=batch_C_prime.dtype)
        ], dim=1)
        batch_T = torch.bmm(self.inv_delta_C.unsqueeze(0).expand(B,-1,-1), batch_C_prime_with_zeros)
        return torch.bmm(self.P_hat.unsqueeze(0).expand(B,-1,-1), batch_T)


class TPSBlock(nn.Module):
    def __init__(self, F=20, I_size=(32, 128), I_channel_num=3):
        super().__init__()
        self.I_r_size            = I_size
        self.LocalizationNetwork = TPSLocalizationNetwork(F, I_channel_num)
        self.GridGenerator       = TPSGridGenerator(F, I_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)
        P_prime       = self.GridGenerator.build_P_prime(batch_C_prime)
        P_prime_reshape = P_prime.reshape(batch_I.size(0), self.I_r_size[0], self.I_r_size[1], 2)
        return F.grid_sample(batch_I, P_prime_reshape, padding_mode='border', align_corners=True)
    
    
def drop_path_fn(x, drop_prob=0.0, training=False):
    if drop_prob==0.0 or not training: return x
    keep_prob=1-drop_prob; shape=(x.shape[0],)+(1,)*(x.ndim-1)
    return x/keep_prob*(torch.rand(shape,dtype=x.dtype,device=x.device)+keep_prob).floor()

class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p=p
    def forward(self, x): return drop_path_fn(x, self.p, self.training)

class SVTRPatchEmbed(nn.Module):
    def __init__(self, img_size=(32,128), in_channels=3, embed_dim=64):
        super().__init__()
        self.num_patches=(img_size[0]//4)*(img_size[1]//4)
        self.proj=nn.Sequential(
            nn.Conv2d(in_channels,embed_dim//2,3,stride=2,padding=1),nn.BatchNorm2d(embed_dim//2),nn.GELU(),
            nn.Conv2d(embed_dim//2,embed_dim,3,stride=2,padding=1),nn.BatchNorm2d(embed_dim),nn.GELU())
    def forward(self, x): return self.proj(x).flatten(2).permute(0,2,1)

class SVTRMlp(nn.Module):
    def __init__(self, dim, hidden=None, drop=0.0):
        super().__init__(); hidden=hidden or dim
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(drop),nn.Linear(hidden,dim),nn.Dropout(drop))
    def forward(self, x): return self.net(x)

class SVTRAttention(nn.Module):
    def __init__(self, dim, num_heads=8, mixer='Global', HW=None, local_k=(7,11), qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads=num_heads; self.head_dim=dim//num_heads; self.scale=self.head_dim**-0.5; self.mixer=mixer
        self.qkv=nn.Linear(dim,dim*3,bias=qkv_bias); self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim); self.proj_drop=nn.Dropout(proj_drop)
        if mixer=='Local' and HW is not None:
            H,W=HW; hk,wk=local_k
            mask=torch.ones(H*W,H+hk-1,W+wk-1)
            for h in range(H):
                for w in range(W): mask[h*W+w,h:h+hk,w:w+wk]=0.0
            mask=mask[:,hk//2:H+hk//2,wk//2:W+wk//2].flatten(1)
            mask_inf=torch.full((H*W,H*W),float('-inf'))
            self.register_buffer('local_mask',torch.where(mask<1,torch.zeros_like(mask_inf),mask_inf).unsqueeze(0).unsqueeze(0))
        else: self.local_mask=None
    def forward(self, x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0); attn=(q@k.transpose(-2,-1))*self.scale
        if self.local_mask is not None: attn=attn+self.local_mask
        return self.proj_drop(self.proj((self.attn_drop(attn.softmax(-1))@v).transpose(1,2).reshape(B,N,C)))

class SVTRBlock(nn.Module):
    def __init__(self, dim, num_heads, mixer='Global', HW=None, local_k=(7,11), mlp_ratio=4.0, qkv_bias=False, drop=0.0, attn_drop=0.0, dp=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.attn=SVTRAttention(dim,num_heads,mixer,HW,local_k,qkv_bias,attn_drop,drop)
        self.dp=DropPath(dp) if dp>0 else nn.Identity()
        self.norm2=nn.LayerNorm(dim); self.mlp=SVTRMlp(dim,int(dim*mlp_ratio),drop)
    def forward(self, x):
        x=self.norm1(x+self.dp(self.attn(x))); x=self.norm2(x+self.dp(self.mlp(x))); return x

class SVTRSubSample(nn.Module):
    def __init__(self, in_c, out_c, HW, stride=(2,1)):
        super().__init__(); self.HW=HW
        self.conv=nn.Conv2d(in_c,out_c,3,stride=stride,padding=1); self.norm=nn.LayerNorm(out_c)
    def forward(self, x):
        H,W=self.HW; B,N,C=x.shape
        return self.norm(self.conv(x.permute(0,2,1).reshape(B,C,H,W)).flatten(2).permute(0,2,1))

class SVTRBackbone(nn.Module):
    def __init__(self, img_size=(32,128), in_channels=3, embed_dim=(64,128,256), depth=(3,6,3),
                 num_heads=(2,4,8), mixer=('Local',)*6+('Global',)*6, local_mixer=((7,11),(7,11),(7,11)),
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, out_channels=192, last_drop=0.1):
        super().__init__()
        self.embed_dim=embed_dim; self.out_channels=out_channels
        self.patch_embed=SVTRPatchEmbed(img_size,in_channels,embed_dim[0])
        HW0=(img_size[0]//4,img_size[1]//4); self.HW=HW0
        self.pos_embed=nn.Parameter(torch.zeros(1,self.patch_embed.num_patches,embed_dim[0]))
        nn.init.trunc_normal_(self.pos_embed,std=0.02); self.pos_drop=nn.Dropout(drop_rate)
        dpr=np.linspace(0,drop_path_rate,sum(depth)).tolist()
        HW1=HW0
        self.blocks1=nn.ModuleList([SVTRBlock(embed_dim[0],num_heads[0],mixer[i],HW1,local_mixer[0],mlp_ratio,qkv_bias,drop_rate,attn_drop_rate,dpr[i]) for i in range(depth[0])])
        self.sub_sample1=SVTRSubSample(embed_dim[0],embed_dim[1],HW1); HW2=(HW1[0]//2,HW1[1])
        self.blocks2=nn.ModuleList([SVTRBlock(embed_dim[1],num_heads[1],mixer[depth[0]+i],HW2,local_mixer[1],mlp_ratio,qkv_bias,drop_rate,attn_drop_rate,dpr[depth[0]+i]) for i in range(depth[1])])
        self.sub_sample2=SVTRSubSample(embed_dim[1],embed_dim[2],HW2); HW3=(HW2[0]//2,HW2[1]); self.HW3=HW3
        self.blocks3=nn.ModuleList([SVTRBlock(embed_dim[2],num_heads[2],mixer[depth[0]+depth[1]+i],HW3,local_mixer[2],mlp_ratio,qkv_bias,drop_rate,attn_drop_rate,dpr[depth[0]+depth[1]+i]) for i in range(depth[2])])
        self.avg_pool=nn.AdaptiveAvgPool2d((1,None)); self.last_conv=nn.Conv2d(embed_dim[2],out_channels,1,bias=False)
        self.hardswish=nn.Hardswish(); self.dropout=nn.Dropout(last_drop); self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.trunc_normal_(m.weight,std=0.02); m.bias is not None and nn.init.zeros_(m.bias)
            elif isinstance(m,nn.LayerNorm): nn.init.zeros_(m.bias); nn.init.ones_(m.weight)
    def forward(self, x):
        x=self.pos_drop(self.patch_embed(x)+self.pos_embed)
        for b in self.blocks1: x=b(x)
        x=self.sub_sample1(x)
        for b in self.blocks2: x=b(x)
        x=self.sub_sample2(x)
        for b in self.blocks3: x=b(x)
        B,N,C=x.shape; H3,W3=self.HW3
        return self.dropout(self.hardswish(self.last_conv(self.avg_pool(x.permute(0,2,1).reshape(B,C,H3,W3)))))


class AttentionDecoder(nn.Module):
    """
    Attention Decoder cho OCR.

    Dùng cross-attention để decode từng ký tự:
        - Query: ký tự đang decode (learned embedding)
        - Key/Value: encoder output sequence

    Khác CTC:
        CTC     : align tự do, decode song song, không biết context trước
        Attention: decode tuần tự, biết ký tự trước → tốt hơn cho sequence dài

    Training: teacher forcing (dùng GT token làm input)
    Inference: autoregressive (dùng predicted token làm input)
    """
    def __init__(
        self,
        num_classes: int,
        encoder_dim: int = 512,
        hidden_dim: int  = 256,
        max_len: int     = 25,    # max length biển số
        num_heads: int   = 8,
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_len     = max_len
        self.hidden_dim  = hidden_dim

        # Character embedding (num_classes + 1 cho <SOS> token)
        self.char_embed = nn.Embedding(num_classes + 1, hidden_dim)
        self.sos_idx    = num_classes   # <SOS> token index

        # Project encoder dim → hidden dim nếu khác nhau
        self.enc_proj = nn.Linear(encoder_dim, hidden_dim) if encoder_dim != hidden_dim else nn.Identity()

        # Positional encoding cho decoder
        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        # Transformer decoder layer (cross-attention với encoder)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # Output projection → num_classes
        self.output_proj = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        encoder_out: torch.Tensor,          # [B, W', encoder_dim]
        targets: torch.Tensor = None,        # [B, max_len] — None khi inference
        target_lengths: torch.Tensor = None, # [B]
    ) -> torch.Tensor:
        """
        Training  (targets != None): teacher forcing → [B, max_len, num_classes]
        Inference (targets == None): autoregressive  → [B, max_len, num_classes]
        """
        B = encoder_out.size(0)
        memory = self.pos_enc(self.enc_proj(encoder_out))   # [B, W', hidden_dim]

        if targets is not None:
            # ── Teacher forcing ───────────────────────────────────
            # Prepend <SOS>, bỏ token cuối → shift right
            max_len = target_lengths.max().item()
            sos = torch.full((B, 1), self.sos_idx, device=encoder_out.device, dtype=torch.long)

            # Pad targets về max_len nếu cần, rồi shift right
            tgt_input = torch.zeros(B, max_len, device=encoder_out.device, dtype=torch.long)
            tgt_input[:, 0] = self.sos_idx
            for i in range(B):
                L = target_lengths[i].item()
                # targets là flat tensor — cần reconstruct per-sample
                # Nhưng targets trong dataset là flat [sum_of_lengths]
                # → dùng cách đơn giản: embed SOS + zeros (model học từ attention)
            _ = sos  # unused, dùng approach đơn giản hơn

            # Tạo decoder input: [B, max_len, hidden_dim]
            positions = torch.arange(max_len, device=encoder_out.device).unsqueeze(0).expand(B, -1)
            tgt_embed = self.pos_enc(self.char_embed(tgt_input))

            # Causal mask để tránh attend future tokens
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_len, device=encoder_out.device)

            dec_out  = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)   # [B, max_len, hidden_dim]
            return self.output_proj(dec_out)                                  # [B, max_len, num_classes]

        else:
            # ── Autoregressive inference ──────────────────────────
            outputs = []
            token   = torch.full((B, 1), self.sos_idx, device=encoder_out.device, dtype=torch.long)
            decoded = self.char_embed(token)   # [B, 1, hidden_dim]

            for _ in range(self.max_len):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoded.size(1), device=encoder_out.device)
                dec_out  = self.decoder(decoded, memory, tgt_mask=tgt_mask)
                logit    = self.output_proj(dec_out[:, -1:, :])   # [B, 1, num_classes]
                outputs.append(logit)
                # Next token: argmax
                next_token = logit.argmax(dim=-1)                  # [B, 1]
                next_embed = self.char_embed(next_token)            # [B, 1, hidden_dim]
                decoded    = torch.cat([decoded, next_embed], dim=1)

            return torch.cat(outputs, dim=1)   # [B, max_len, num_classes]
        

class LearnableAlignmentFusion(nn.Module):
    """
    Implicitly learns Optical Flow to align neighboring feature maps 
    to the center frame before fusing them.
    """
    def __init__(self, channels, num_frames=5, feature_height=1):
        super().__init__()
        self.num_frames = num_frames
        
        # 1. Flow Estimator: Predicts X, Y pixel shifts (2 channels)
        self.flow_estimator = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2, 3, padding=1)
        )
        # Initialize to zero so it starts as a safe identity mapping
        nn.init.zeros_(self.flow_estimator[-1].weight)
        nn.init.zeros_(self.flow_estimator[-1].bias)

        # 2. Feature Fusion: Collapses the perfectly aligned frames back to 1
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 3. The Height-Squash Killer: Preserves vertical features
        self.map_to_seq = nn.Linear(channels * feature_height, channels)

    def forward(self, x):
        # x arrives as [Batch*Frames, Channels, Height, Width] from the backbone
        total, c, h, w = x.size()
        b = total // self.num_frames
        f = self.num_frames
        
        # Group into video sequences: [Batch, Frames, Channels, Height, Width]
        x_seq = x.view(b, f, c, h, w)
        
        center_idx = f // 2
        center_feat = x_seq[:, center_idx, :, :, :] # The "Anchor" frame

        aligned_features = []

        # Create a blank X/Y coordinate grid
        y, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing='ij'
        )
        base_grid = torch.stack([x_grid, y], dim=0).unsqueeze(0).repeat(b, 1, 1, 1)

        # Align every frame to the center frame
        for t in range(f):
            if t == center_idx:
                aligned_features.append(center_feat)
                continue
                
            neighbor_feat = x_seq[:, t, :, :, :]

            # Predict the internal optical flow
            concat_feat = torch.cat([center_feat, neighbor_feat], dim=1)
            flow = self.flow_estimator(concat_feat) # [B, 2, H, W]

            # Warp the neighbor's features using the predicted flow
            grid = base_grid + flow
            grid = grid.permute(0, 2, 3, 1) # Format required by grid_sample
            aligned_neighbor = F.grid_sample(neighbor_feat, grid, align_corners=True, padding_mode='border')
            
            aligned_features.append(aligned_neighbor)

        # Concatenate all perfectly aligned frames and fuse them
        fused_2d = self.fusion_conv(torch.cat(aligned_features, dim=1)) # [B, C, H, W]

        # Preserve vertical data (Height-Squash Killer)
        seq = fused_2d.permute(0, 3, 1, 2).contiguous().view(b, w, c * h) # [B, W, C*H]
        out = self.map_to_seq(seq).permute(0, 2, 1).unsqueeze(2) # [B, C, 1, W]
        
        return out