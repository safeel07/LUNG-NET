import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import MONAI components
try:
    from monai.networks.nets import DenseNet121
    MONAI_NET_AVAILABLE = True
except ImportError:
    MONAI_NET_AVAILABLE = False


class TabularProjectionBlock(nn.Module):
    """
    Projection embedding block for Clinical and Genetic data payloads.
    Translates discrete genetic IntEnums and continuous clinical metrics
    into a unified 256-dimensional tabular latent representation space.
    """
    def __init__(self):
        super().__init__()
        # Categories: 0=WT, 1=Mutant, 2=Unknown (3 embeddings mapped to 16d)
        self.egfr_emb = nn.Embedding(3, 16)
        self.kras_emb = nn.Embedding(3, 16)
        self.alk_emb = nn.Embedding(3, 16)
        
        # Clinical parameters projection layer: Age, Pack-Years -> 208d
        self.clin_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 208),
            nn.ReLU()
        )
        
        # Unified projection head (16*3 + 208 = 256d)
        self.unified_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
    def forward(self, x):
        # x is assumed to be shape (B, 5): [Age, Pack-Years, EGFR, KRAS, ALK]
        clin_vars = x[:, :2]
        egfr_cat = x[:, 2].long()
        kras_cat = x[:, 3].long()
        alk_cat = x[:, 4].long()
        
        # Extract embeddings
        egfr_feats = self.egfr_emb(egfr_cat)
        kras_feats = self.kras_emb(kras_cat)
        alk_feats = self.alk_emb(alk_cat)
        
        # Project clinical features
        clin_feats = self.clin_proj(clin_vars)
        
        # Concat: shape (B, 256)
        fused_tabular = torch.cat([clin_feats, egfr_feats, kras_feats, alk_feats], dim=1)
        return self.unified_proj(fused_tabular)


class AttentionGatedMultimodalFusion(nn.Module):
    """
    Attention-Gated Cross-Modal Fusion layer.
    Computes direct multi-head cross-attention across Vision representations
    and Tabular representations. Clinical features gate visual features.
    """
    def __init__(self, img_dim=1024, tab_dim=256, fused_dim=512):
        super().__init__()
        # Query projection from tabular (query guides the attention focus)
        self.query_proj = nn.Linear(tab_dim, fused_dim)
        # Key & Value projections from high-capacity imaging backbone
        self.key_proj = nn.Linear(img_dim, fused_dim)
        self.value_proj = nn.Linear(img_dim, fused_dim)
        
        # Multi-Head Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4, batch_first=True)
        
        # Normalization and output project
        self.norm1 = nn.LayerNorm(fused_dim)
        self.norm2 = nn.LayerNorm(fused_dim)
        
    def forward(self, img_feats, tab_feats):
        # Shape of img_feats: (B, 1024), tab_feats: (B, 256)
        # Project and unsqueeze to sequential format (B, SeqLen, Dim) -> (B, 1, fused_dim)
        q = self.query_proj(tab_feats).unsqueeze(1)
        k = self.key_proj(img_feats).unsqueeze(1)
        v = self.value_proj(img_feats).unsqueeze(1)
        
        # Execute Cross Attention: Tabular query attends to Image keys/values
        attn_out, attn_weights = self.cross_attn(query=q, key=k, value=v)
        
        # Reshape back to vector format and add residual link from tabular
        fused = attn_out.squeeze(1)
        fused = self.norm1(fused + q.squeeze(1))
        
        return fused, attn_weights


class AttentionGatedFusionNet(nn.Module):
    """
    Enterprise Multimodal attention-gated neural classifier.
    Combines MONAI DenseNet121, Tabular projections, and Multi-head cross-attention.
    """
    def __init__(self):
        super().__init__()
        
        # 1. Vision branch: MONAI 3D DenseNet-121
        if MONAI_NET_AVAILABLE:
            self.image_backbone = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=1024
            )
            # Remove only final linear layer, preserving ReLU, pooling, and flattening layers
            self.image_backbone.class_layers[-1] = nn.Identity()
        else:
            # Fallback block if MONAI is missing during bootstrap checks
            self.image_backbone = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(),
                nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(64, 1024),
                nn.ReLU()
            )
            
        # 2. Tabular branch: Clinical Projection Embedding Block
        self.tabular_branch = TabularProjectionBlock()
        
        # 3. Fusion layer: Gated Multi-Head Cross Attention
        self.fusion_layer = AttentionGatedMultimodalFusion(img_dim=1024, tab_dim=256, fused_dim=512)
        
        # 4. Deep MLP Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1) # Raw logit output
        )
        
        # Explainability activation and gradient trackers
        self.grad_cam_gradients = None
        self.grad_cam_activations = None
        self.hook_handle_forward = None
        self.hook_handle_backward = None
        
        # Register explainability hooks dynamically
        self.register_grad_cam_hooks()
        
    def forward_hook(self, module, input, output):
        self.grad_cam_activations = output
        
    def backward_hook(self, module, grad_input, grad_output):
        self.grad_cam_gradients = grad_output[0]
        
    def register_grad_cam_hooks(self):
        """
        Dynamically locates the final convolutional layer of the MONAI DenseNet
        or fallback vision block and attaches activation and gradient tracking hooks.
        """
        target_layer = None
        
        # Traverse network to locate the very last Conv3D layer in the vision branch
        for _, sub_module in self.image_backbone.named_modules():
            if isinstance(sub_module, nn.Conv3d):
                target_layer = sub_module
                
        if target_layer is not None:
            # Register hooks to save target activations and backprop gradients
            self.hook_handle_forward = target_layer.register_forward_hook(self.forward_hook)
            self.hook_handle_backward = target_layer.register_full_backward_hook(self.backward_hook)
            
    def forward(self, img, tab):
        img_feats = self.image_backbone(img) # (B, 1024)
        tab_feats = self.tabular_branch(tab) # (B, 256)
        
        # Gated Multi-Head Cross Attention Fusion
        fused_vector, attn_weights = self.fusion_layer(img_feats, tab_feats)
        
        # Classification projection
        logits = self.classifier(fused_vector)
        return logits


def generate_densenet_gradcam(model, img_tensor, tab_tensor):
    """
    Computes a deterministic 3D Grad-CAM spatial activation heatmap.
    Tracks visual gradients matching clinical predictions back to 3D image coordinates.
    """
    model.eval()
    
    with torch.enable_grad():
        img_input = img_tensor.clone().detach().requires_grad_(True)
        tab_input = tab_tensor.clone().detach()
        
        # Forward pass
        logits = model(img_input, tab_input)
        
        # Reset model gradients
        model.zero_grad()
        
        # Backward pass on the logit output
        logits.backward()
        
        # Read saved activations and gradients
        gradients = model.grad_cam_gradients
        activations = model.grad_cam_activations
        
        if gradients is None or activations is None:
            # Uniform fallback if gradient tracking failed
            return np.ones((64, 64, 64), dtype=np.float32) * 0.1
            
        # Global average pool the gradients along depth, height, width
        weights = torch.mean(gradients, dim=[2, 3, 4], keepdim=True)
        
        # Compute weighted channel activations
        weighted_act = torch.sum(activations * weights, dim=1).squeeze(0)
        
        # Apply ReLU to retain only features positive to the output
        heatmap = F.relu(weighted_act)
        
        # Upscale heatmap using trilinear interpolation back to original tensor size (64, 64, 64)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(64, 64, 64), mode='trilinear', align_corners=False)
        heatmap = heatmap.squeeze(0).squeeze(0)
        
        # Convert to numpy and normalize
        heatmap_np = heatmap.detach().cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np = heatmap_np / max_val
            
        return heatmap_np
