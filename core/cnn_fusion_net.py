try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    
    class DummyModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return self
        def eval(self, *args, **kwargs):
            return self
        def train(self, *args, **kwargs):
            return self
        def to(self, *args, **kwargs):
            return self
        def cuda(self, *args, **kwargs):
            return self
        def cpu(self, *args, **kwargs):
            return self
        def modules(self, *args, **kwargs):
            return []
        def parameters(self, *args, **kwargs):
            return []
        def state_dict(self, *args, **kwargs):
            return {}
        def load_state_dict(self, *args, **kwargs):
            return self
        def register_forward_hook(self, *args, **kwargs):
            pass
        def register_full_backward_hook(self, *args, **kwargs):
            pass

    class DummyNN:
        Module = DummyModule
        def __getattr__(self, name):
            class DummyCallableClass(DummyModule):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                def __getattr__(self, item):
                    return lambda *args, **kwargs: None
            return DummyCallableClass
            
    nn = DummyNN()
    
    class DummyF:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    F = DummyF()

try:
    from monai.networks.nets import DenseNet121
    MONAI_NET_AVAILABLE = True
except Exception:
    MONAI_NET_AVAILABLE = False


class TabularProjectionBlock(nn.Module):
    """
    Genomics & continuous demographics co-embedding stream.
    Creates structured embedding representations for EGFR, KRAS, and ALK genotypes
    and merges them with continuous scaled parameters to project a 256d vector.
    """
    def __init__(self):
        super().__init__()
        # 3 categorical states: 0=Wild-Type, 1=Mutant, 2=Unknown
        self.egfr_embedding = nn.Embedding(num_embeddings=3, embedding_dim=16)
        self.kras_embedding = nn.Embedding(num_embeddings=3, embedding_dim=16)
        self.alk_embedding = nn.Embedding(num_embeddings=3, embedding_dim=16)
        
        self.continuous_projection = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 208),
            nn.ReLU()
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.12)
        )

    def forward(self, tabular_input):
        # Input tensor structure: [Age, Smoking Pack-Years, EGFR, KRAS, ALK]
        continuous_features = tabular_input[:, :2]
        egfr_status = tabular_input[:, 3].long()
        kras_status = tabular_input[:, 4].long()
        alk_status = tabular_input[:, 5].long() if tabular_input.shape[1] > 5 else tabular_input[:, 4].long()
        
        egfr_feat = self.egfr_embedding(egfr_status)
        kras_feat = self.kras_embedding(kras_status)
        alk_feat = self.alk_embedding(alk_status)
        
        continuous_feat = self.continuous_projection(continuous_features)
        
        merged_vector = torch.cat([continuous_feat, egfr_feat, kras_feat, alk_feat], dim=1)
        return self.output_projection(merged_vector)


class CrossModalityAttentionGate(nn.Module):
    """
    Multi-Head Cross-Attention Layer gating.
    Tabular features serve as Query vectors to attend and filter 3D vision Key/Value maps.
    """
    def __init__(self, vision_dim=1024, tabular_dim=256, projection_dim=512):
        super().__init__()
        self.query_proj = nn.Linear(tabular_dim, projection_dim)
        self.key_proj = nn.Linear(vision_dim, projection_dim)
        self.value_proj = nn.Linear(vision_dim, projection_dim)
        
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=4,
            batch_first=True
        )
        self.layer_norm_1 = nn.LayerNorm(projection_dim)
        self.layer_norm_2 = nn.LayerNorm(projection_dim)

    def forward(self, vision_features, tabular_features):
        # Create token representations
        q = self.query_proj(tabular_features).unsqueeze(1) # (B, 1, projection_dim)
        k = self.key_proj(vision_features).unsqueeze(1) # (B, 1, projection_dim)
        v = self.value_proj(vision_features).unsqueeze(1) # (B, 1, projection_dim)
        
        # Calculate scaled dot-product cross-attention
        attention_output, attention_weights = self.multihead_attention(
            query=q, 
            key=k, 
            value=v
        )
        
        # Merge residual patterns
        fused_representation = attention_output.squeeze(1)
        fused_representation = self.layer_norm_1(fused_representation + q.squeeze(1))
        
        return fused_representation, attention_weights


class AttentionGatedFusionNet(nn.Module):
    """
    Convolutional Multi-Modal late-fusion diagnostic system.
    Combines MONAI 3D DenseNet-121, co-embeddings, and Multi-Head Cross-Attention.
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
                nn.Conv3d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d(2),
                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(128, 1024)
            )
            
        # 2. Tabular branch: Genomic-clinical projections
        self.tabular_backbone = TabularProjectionBlock()
        
        # 3. Multimodal late-fusion cross-attention block
        self.fusion_layer = CrossModalityAttentionGate(
            vision_dim=1024,
            tabular_dim=256,
            projection_dim=512
        )
        
        # 4. Deep MLP Classification Head
        self.classifier_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1) # Raw logit output
        )
        
        # Explainability tracking parameters
        self.hook_gradients = None
        self.hook_activations = None
        self.h_forward = None
        self.h_backward = None
        
        # Register explainability hook
        self.register_grad_cam_hooks()

    def forward_hook(self, module, input, output):
        self.hook_activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.hook_gradients = grad_output[0]

    def register_grad_cam_hooks(self):
        """
        Dynamically locates the final Convolutional layer in MONAI DenseNet121 block
        to register forward and backward hooks for Grad-CAM.
        """
        target_conv = None
        
        if MONAI_NET_AVAILABLE:
            # Traverse MONAI DenseNet modules to identify final convolutional layer
            for submodule in self.image_backbone.modules():
                if isinstance(submodule, nn.Conv3d):
                    target_conv = submodule
        else:
            # Revert to standard fallback convolutional layers
            for submodule in self.image_backbone.modules():
                if isinstance(submodule, nn.Conv3d):
                    target_conv = submodule
                    
        if target_conv is not None:
            self.h_forward = target_conv.register_forward_hook(self.forward_hook)
            self.h_backward = target_conv.register_full_backward_hook(self.backward_hook)
        else:
            print("[WARN] Could not register diagnostic Grad-CAM hooks: Conv3D layer not found.")

    def forward(self, img_input, tab_input):
        # 1. Vision representations: (B, 1024)
        img_feats = self.image_backbone(img_input)
        
        # 2. Tabular representations: (B, 256)
        tab_feats = self.tabular_backbone(tab_input)
        
        # 3. Attention-Gated Multi-Modal Fusion
        fused_vector, attn_weights = self.fusion_layer(img_feats, tab_feats)
        
        # 4. Score Logits
        logits = self.classifier_head(fused_vector)
        return logits


def generate_densenet_gradcam(model, img_tensor, tab_tensor):
    """
    Generates a deterministic 3D Grad-CAM visualization from convolutional channels.
    """
    model.eval()
    
    with torch.enable_grad():
        img_input = img_tensor.clone().detach().requires_grad_(True)
        tab_input = tab_tensor.clone().detach()
        
        # Forward pass execution
        logits = model(img_input, tab_input)
        
        # Clean model gradients
        model.zero_grad()
        
        # Backprop logit targets
        logits.backward()
        
        # Extract gradient maps and activation maps
        gradients = model.hook_gradients
        activations = model.hook_activations
        
        if gradients is None or activations is None:
            # Clean uniform fallback
            return np.ones((64, 64, 64), dtype=np.float32) * 0.1
            
        # Compute channel-wise average pooled gradients
        weights = torch.mean(gradients, dim=[2, 3, 4], keepdim=True)
        
        # Compute weighted sum of spatial activations
        weighted_act = torch.sum(activations * weights, dim=1).squeeze(0)
        
        # Filter negative activations using ReLU
        heatmap = F.relu(weighted_act)
        
        # Upscale heatmap sequence back to volumetric shape (64, 64, 64)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = F.interpolate(heatmap, size=(64, 64, 64), mode='trilinear', align_corners=False)
        heatmap = heatmap.squeeze(0).squeeze(0)
        
        import numpy as np
        heatmap_np = heatmap.detach().cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np = heatmap_np / max_val
            
        return heatmap_np
