import os
import math
import random
import time
import json
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

# CRITICAL FOR .PY SCRIPTS: Forces matplotlib to save images instead of trying to open GUI windows
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Linear, Dropout, LayerNorm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment 
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from timm.data import Mixup

from fvcore.nn import FlopCountAnalysis, parameter_count

import sys
# Force stdout and stderr to flush immediately, even when redirected to a file
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import warnings
# Ignore standard UserWarnings to keep the log file clean
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Reproducibility / device
# ---------------------------
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 4090 gpu optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# FLOPs calculation function
def count_flops_fvcore(model, img_size=32):
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    flops = FlopCountAnalysis(model, dummy_input)
    total_flops = flops.total()
    params = parameter_count(model)
    return {
        'params': params[''],
        'params_str': f"{params['']:,}",
        'flops': total_flops,
        'flops_str': f"{total_flops/1e9:.2f} GFLOPs"
    }

# ---------------------------
# Model components (base)
# ---------------------------
class PatchExtractor(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3): # Fixed dunder
        super().__init__() # Fixed dunder
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size doesn't match model"
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, -1, self.in_channels * self.patch_size * self.patch_size)
        return x

class GraphConstructor(nn.Module):
    def __init__(self, grid_size, connectivity=8): # Fixed dunder
        super().__init__() # Fixed dunder
        self.grid_size = grid_size
        self.connectivity = connectivity
        self.register_buffer("spatial_dist", self.compute_spatial_distances())
        self.register_buffer("candidate_mask", self.spatial_dist > 1)
        
    def compute_spatial_distances(self):
        g = self.grid_size
        coords = torch.arange(g)
        row_coords = coords.view(-1, 1).expand(g, g).reshape(-1)
        col_coords = coords.view(1, -1).expand(g, g).reshape(-1)
        row_diff = (row_coords.view(-1, 1) - row_coords.view(1, -1)).abs()
        col_diff = (col_coords.view(-1, 1) - col_coords.view(1, -1)).abs()
        if self.connectivity == 4:
            dist = row_diff + col_diff
        else:
            dist = torch.max(row_diff, col_diff)
        return dist.long()
    
class SpatialEncodingShared(nn.Module):
    def __init__(self, max_distance, num_heads): # Fixed dunder
        super().__init__() # Fixed dunder
        self.max_distance = int(max_distance)
        self.num_heads = int(num_heads)
        self.distance_bias = nn.Parameter(torch.zeros(self.max_distance + 1))
        self.virtual_bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.distance_bias, std=0.02) # Fixed underscore
        nn.init.normal_(self.virtual_bias, std=0.02) # Fixed underscore

    def compute_bias(self, dist_matrix):
        B, N, _ = dist_matrix.shape
        vmask = dist_matrix < 0
        dist_clamped = torch.clamp(dist_matrix, 0, self.max_distance)
        db = self.distance_bias[dist_clamped] 
        db = torch.where(vmask, self.virtual_bias, db)
        return db.unsqueeze(1)

    def forward(self, attn_scores, dist_matrix=None, precomputed_bias=None):
        B, H, N, _ = attn_scores.shape
        if precomputed_bias is not None:
            bias_to_add = precomputed_bias
        else:
            if dist_matrix.dim() == 2:
                dist = dist_matrix.unsqueeze(0)
            else:
                dist = dist_matrix
            bias_to_add = self.compute_bias(dist)
            if bias_to_add.shape[0] == 1 and B > 1:
                bias_to_add = bias_to_add.expand(B, -1, -1, -1)
        return attn_scores + bias_to_add.to(attn_scores.dtype)

class BucketedCosineSimilarityBias(nn.Module):
    """
    Bucketed cosine similarity bias with LEARNABLE thresholds.

    Soft bucketing via sigmoid transitions:
      sim << t_low   → bucket 0 (fixed zero, baseline)
      t_low < sim < t_high → bucket 1 (learnable medium bias)
      sim >> t_high  → bucket 2 (learnable strong bias)
      self-connection → zeroed out explicitly

    Thresholds t_low, t_high are nn.Parameters, initialized at 0.4 and 0.7.
    Ordering (t_low < t_high) is enforced via min/max in forward.
    """

    SHARPNESS = 20.0  # Fixed. Higher → closer to hard bucketing. Do NOT make learnable.

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

        # --- Learnable bucket biases (only 2; buckets 0 and 3 are fixed zero) ---
        self.learnable_biases = nn.Parameter(torch.zeros(2))
        nn.init.normal_(self.learnable_biases, std=0.02)

        # --- Learnable thresholds (initialized at original hardcoded values) ---
        self.thresh_low = nn.Parameter(torch.tensor(0.4))
        self.thresh_high = nn.Parameter(torch.tensor(0.7))

    def get_bucket_weights(self):
        """For diagnostics: return the 4-element weight vector."""
        w = torch.zeros(4, device=self.learnable_biases.device, dtype=self.learnable_biases.dtype)
        w[1] = self.learnable_biases[0]
        w[2] = self.learnable_biases[1]
        return w

    def get_learned_thresholds(self):
        """For diagnostics: return the current threshold values (ordered)."""
        with torch.no_grad():
            t_low = torch.min(self.thresh_low, self.thresh_high).item()
            t_high = torch.max(self.thresh_low, self.thresh_high).item()
        return t_low, t_high

    def compute_bias_from_patches(self, patches):
        """
        patches: (B, Np, D) — pure projected patch embeddings (no CLS, no pos, no centrality)
        Returns: (B, 1, N_full, N_full) with CLS rows/cols zeroed.
        """
        B, Np, D = patches.shape
        if Np <= 0:
            N_full = 1
            return torch.zeros(B, 1, N_full, N_full, device=patches.device, dtype=patches.dtype)

        # Cosine similarity
        patches_norm = F.normalize(patches, dim=-1)
        sim = torch.einsum('bnd,bmd->bnm', patches_norm, patches_norm)  # (B, Np, Np)

        # Enforce threshold ordering: t_low <= t_high
        t_low = torch.min(self.thresh_low, self.thresh_high)
        t_high = torch.max(self.thresh_low, self.thresh_high)

        # --- Soft bucket membership via sigmoid ---
        # w_low  ≈ 1 when sim < t_low,   ≈ 0 when sim > t_low
        # w_high ≈ 1 when sim > t_high,  ≈ 0 when sim < t_high
        # w_mid  ≈ 1 when t_low < sim < t_high
        w_low = torch.sigmoid(self.SHARPNESS * (t_low - sim))    # bucket 0 membership
        w_high = torch.sigmoid(self.SHARPNESS * (sim - t_high))  # bucket 2 membership
        w_mid = (1.0 - w_low - w_high).clamp(min=0.0)            # bucket 1 membership

        # Weighted bias (bucket 0 contributes zero, so only mid and high matter)
        bias_patch = w_mid * self.learnable_biases[0] + w_high * self.learnable_biases[1]
        # shape: (B, Np, Np)

        # Zero out self-connections (diagonal)
        eye_mask = torch.eye(Np, device=sim.device, dtype=torch.bool).unsqueeze(0)
        bias_patch = bias_patch.masked_fill(eye_mask, 0.0)

        # Build full matrix with CLS row/col = 0
        N_full = Np + 1
        bias_full = torch.zeros(B, 1, N_full, N_full, device=sim.device, dtype=bias_patch.dtype)
        bias_full[:, :, 1:, 1:] = bias_patch.unsqueeze(1)  # (B, 1, Np, Np)

        return bias_full  # (B, 1, N_full, N_full)

    def forward(self, attn_scores, x_with_cls=None, precomputed_bias=None):
        """Called inside MultiHeadAttention. Bias is already (B, H, N, N) from main model."""
        if precomputed_bias is not None:
            return attn_scores + precomputed_bias.to(attn_scores.dtype)
        return attn_scores

class MultiHeadAttentionWithSpatialAndCosine(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, spatial_encoder, cosine_encoder): # Fixed dunder
        super().__init__() # Fixed dunder
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = (self.head_dim) ** -0.5
        self.spatial = spatial_encoder
        self.cosine = cosine_encoder

    def forward(self, x, dist_matrix=None, spatial_bias=None, cosine_bias=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.spatial(attn, dist_matrix=dist_matrix, precomputed_bias=spatial_bias)
        attn = self.cosine(attn, x_with_cls=None, precomputed_bias=cosine_bias)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class TransformerEncoderLayerWithSpatialAndCosine(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, spatial_encoder, cosine_encoder, dim_feedforward=None): # Fixed dunder
        super().__init__() # Fixed dunder
        if dim_feedforward is None:
            dim_feedforward = embed_dim * 4
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionWithSpatialAndCosine(embed_dim, num_heads, dropout=dropout,
                                                           spatial_encoder=spatial_encoder,
                                                           cosine_encoder=cosine_encoder)
        self.dropout1 = Dropout(dropout)
        self.norm2 = LayerNorm(embed_dim)
        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, embed_dim)
        self.dropout2 = Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, dist_matrix=None, spatial_bias=None, cosine_bias=None):
        y = self.norm1(x)
        y = self.attn(y, dist_matrix=dist_matrix, spatial_bias=spatial_bias, cosine_bias=cosine_bias)
        x = x + self.dropout1(y)
        y2 = self.norm2(x)
        ff = self.linear2(self.dropout(self.activation(self.linear1(y2))))
        x = x + self.dropout2(ff)
        return x

class ViTGraphormerDynamic(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=191, num_heads=3, num_layers=12, connectivity=8, dropout=0.1): # Fixed dunder
        super().__init__() # Fixed dunder
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.connectivity = connectivity

        self.patch_extractor = PatchExtractor(img_size=img_size, patch_size=patch_size, in_channels=in_channels)
        patch_dim = in_channels * patch_size * patch_size
        self.patch_proj = Linear(patch_dim, embed_dim)

        initial_values = torch.tensor([0.03, 0.05, 0.08])
        self.degree_scalars = nn.Parameter(initial_values)
        deg = self._compute_node_degrees()
        if connectivity == 4:
            deg_to_class = {2: 0, 3: 1, 4: 2}
        else:
            deg_to_class = {3: 0, 5: 1, 8: 2}
        degree_values = deg.tolist()
        degree_class_indices = torch.tensor([deg_to_class[int(d)] for d in degree_values], dtype=torch.long)
        self.register_buffer("degree_class_buffer", degree_class_indices)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02) # Fixed underscore
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.graph_constructor = GraphConstructor(grid_size=self.grid_size, connectivity=connectivity)

        if connectivity == 8:
            self.max_distance = self.grid_size - 1
        else:
            self.max_distance = 2 * (self.grid_size - 1)

        self.shared_spatial_encoder = SpatialEncodingShared(self.max_distance, num_heads)
        self.cosine_encoder = BucketedCosineSimilarityBias(num_heads)

        self.spatial_alpha = nn.Parameter(torch.ones(num_layers, num_heads))
        self.cosine_alpha = nn.Parameter(torch.full((num_layers, num_heads), 5.0))

        layer_embed_dim = embed_dim + 1
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithSpatialAndCosine(
                embed_dim=layer_embed_dim, num_heads=num_heads, dropout=dropout,
                spatial_encoder=self.shared_spatial_encoder, cosine_encoder=self.cosine_encoder,
                dim_feedforward=4 * layer_embed_dim
            ) for _ in range(num_layers)
        ])

        self.norm = LayerNorm(layer_embed_dim)
        self.head = Linear(layer_embed_dim, num_classes)

    def _compute_node_degrees(self):
        g = self.grid_size
        degrees = []
        for r in range(g):
            for c in range(g):
                if self.connectivity == 4:
                    if (r in (0, g-1)) and (c in (0, g-1)): deg = 2
                    elif (r in (0, g-1)) or (c in (0, g-1)): deg = 3
                    else: deg = 4
                else:
                    if (r in (0, g-1)) and (c in (0, g-1)): deg = 3
                    elif (r in (0, g-1)) or (c in (0, g-1)): deg = 5
                    else: deg = 8
                degrees.append(deg)
        return torch.tensor(degrees, dtype=torch.long)

    def forward(self, x):
        B = x.shape[0]
        device = x.device
        x = self.patch_extractor(x)
        x_proj = self.patch_proj(x)
        cosine_bias_full = self.cosine_encoder.compute_bias_from_patches(x_proj)

        deg_classes = self.degree_class_buffer.to(x.device)
        degree_vals = self.degree_scalars[deg_classes]
        degree_expanded = degree_vals.view(1, -1, 1).expand(B, -1, -1)
        x_aug = torch.cat([x_proj, degree_expanded], dim=-1)

        cls = self.cls_token.expand(B, -1, -1).to(device)
        cls = torch.cat([cls, torch.zeros(B, 1, 1, device=x.device)], dim=-1)
        x = torch.cat([cls, x_aug], dim=1)

        pos = torch.cat([self.pos_embed, torch.zeros_like(self.pos_embed[:, :, :1])], dim=-1)
        x = x + pos.to(device)

        patch_spatial = self.graph_constructor.spatial_dist.unsqueeze(0).expand(B, -1, -1)
        dist_full = self._extend_distances_with_cls(patch_spatial, device)
        spatial_bias_full = self.shared_spatial_encoder.compute_bias(dist_full)

        for l, layer in enumerate(self.layers):
            alpha_sp = self.spatial_alpha[l].view(1, -1, 1, 1) 
            spatial_bias_l = alpha_sp * spatial_bias_full
            alpha_cos = self.cosine_alpha[l].view(1, -1, 1, 1)
            cosine_bias_l = alpha_cos * cosine_bias_full
            x = layer(x, dist_matrix=None, spatial_bias=spatial_bias_l, cosine_bias=cosine_bias_l)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits
    
    def _extend_distances_with_cls(self, patch_dist, device):
        B = patch_dist.shape[0]
        Np = self.num_patches
        full_N = Np + 1
        dist_full = torch.zeros(B, full_N, full_N, dtype=torch.long, device=device)
        dist_full[:, 1:, 1:] = patch_dist
        dist_full[:, 0, 1:] = -1
        dist_full[:, 1:, 0] = -1
        dist_full[:, 0, 0] = 0
        return dist_full

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0): # Fixed dunder
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True

# ==========================================
# CELL 1: TRAINING (Unindented to preserve global scope for subsequent cells)
# ==========================================
img_size = 32
patch_size = 4
batch_size = 128
num_epochs = 300
lr = 1e-3
weight_decay = 0.05
num_classes = 10

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
transform_train = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomCrop(img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=10),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, inplace=True),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, 
                 switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=num_classes)

model = ViTGraphormerDynamic(
    img_size=img_size, patch_size=patch_size, in_channels=3, num_classes=num_classes,
    embed_dim=191, num_heads=3, num_layers=12, connectivity=8, dropout=0.1
).to(device)

stats = count_flops_fvcore(model)
print(f"fvcore: {stats['params_str']} params, {stats['flops_str']}")

model = torch.compile(model, mode='reduce-overhead')

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

warmup_epochs = 5
total_epochs = num_epochs
warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                            total_iters=warmup_epochs * len(train_loader))
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs) * len(train_loader))
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                        milestones=[warmup_epochs * len(train_loader)])

early_stopper = EarlyStopping(patience=num_epochs, min_delta=1e-3)

best_acc = 0.0
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        orig_labels = labels.clone()
        images, labels = mixup_fn(images, labels)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            logits = model(images)
            loss = criterion(logits, labels)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step() 
        scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = logits.max(1)
        total += labels.size(0)
        correct += preds.eq(orig_labels).sum().item()


    train_loss = running_loss / total
    train_acc = 100. * correct / total

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = logits.max(1) # Fixed Typo from notebook JSON export
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    val_loss = val_loss / val_total
    val_acc = 100. * val_correct / val_total

    print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_svg_main.pth")
        print(f"New best val acc: {best_acc:.2f}% (model saved)")

    early_stopper.step(val_loss)
    if early_stopper.should_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

print(f"Training finished. Best val acc: {best_acc:.2f}%")

t_low, t_high = model.cosine_encoder.get_learned_thresholds()
print(f"Learned thresholds: t_low = {t_low:.4f}, t_high = {t_high:.4f}")
print(f"  (initialized at 0.4 and 0.7)")
print(f"  Shift: Δt_low = {t_low - 0.4:+.4f}, Δt_high = {t_high - 0.7:+.4f}")