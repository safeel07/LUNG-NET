import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImageBranch3D(nn.Module):
    """
    3D CNN feature extractor for simulated or uploaded 3D Lung Nodule ROI volumes.
    Accepts (Batch, 1, 64, 64, 64) and produces a 128-dimensional latent vector.
    """
    def __init__(self):
        super().__init__()
        # Conv block 1: Input 64x64x64 -> Output 32x32x32
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Conv block 2: Input 32x32x32 -> Output 16x16x16
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Conv block 3: Input 16x16x16 -> Output 8x8x8
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Conv block 4: Input 8x8x8 -> Output 8x8x8 (target layer for Grad-CAM)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2) # Final pooled output: 4x4x4
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Placeholders for Grad-CAM
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Target layer for Grad-CAM: conv4 output before final pool
        x = F.relu(self.bn4(self.conv4(x)))
        self.activations = x
        
        # Register backward hook to capture gradients on activation map
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            
        x = self.pool4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) # Flatten to 128 dimensions
        return x


class TabularMLP(nn.Module):
    """
    MLP baseline branch that takes 5 clinical & genetic features:
    [Age, Smoking Pack-Years, EGFR (0/1), KRAS (0/1), ALK (0/1)]
    Outputs a 32-dimensional embedding.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Prevent BN crash for singleton batch sizes during isolated inference
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
        return x


class MultimodalLungNet(nn.Module):
    """
    Late Fusion Multimodal Architecture combining 3D-CNN Image features
    and Tabular MLP features into a unified Risk Stratification Head.
    """
    def __init__(self):
        super().__init__()
        self.image_branch = ImageBranch3D()
        self.tabular_branch = TabularMLP()
        
        # Unified Late Fusion Classification Head
        # Concatenated features dimension = 128 (image) + 32 (tabular) = 160
        self.classifier = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Raw logit output (Sigmoid is applied later)
        )
        
    def forward(self, img, tab):
        img_feats = self.image_branch(img)
        tab_feats = self.tabular_branch(tab)
        
        # Concatenate features along the channel dimension
        fused = torch.cat((img_feats, tab_feats), dim=1)
        logit = self.classifier(fused)
        return logit


def generate_3d_gradcam(model, img_tensor, tab_tensor):
    """
    Generates a functional 3D Grad-CAM heatmap overlay for the 3D Image branch.
    Runs a forward and backward pass, averaging the gradients of the last conv layer
    and combining them with forward activation maps.
    
    Args:
        model: An instance of MultimodalLungNet in eval mode.
        img_tensor: Torch tensor of shape (1, 1, 64, 64, 64).
        tab_tensor: Torch tensor of shape (1, 5).
        
    Returns:
        heatmap_3d: A numpy array of shape (64, 64, 64) with normalized [0, 1] intensities.
    """
    # Ensure gradients are enabled for local Grad-CAM calculation
    with torch.enable_grad():
        img_input = img_tensor.clone().detach().requires_grad_(True)
        tab_input = tab_tensor.clone().detach()
        
        # Forward pass
        logit = model(img_input, tab_input)
        
        # Reset model gradients
        model.zero_grad()
        
        # Backward pass on raw logit
        logit.backward()
        
        # Extract activations and gradients
        gradients = model.image_branch.gradients
        activations = model.image_branch.activations
        
        if gradients is None or activations is None:
            # Return uniform fallback if gradients are unavailable
            return np.ones((64, 64, 64), dtype=np.float32) * 0.1
            
        # Perform global average pooling on 3D gradients
        # Shape of gradients: [1, 128, 8, 8, 8]
        weights = torch.mean(gradients, dim=[2, 3, 4], keepdim=True) # [1, 128, 1, 1, 1]
        
        # Compute weighted sum of activations
        weighted_act = torch.sum(activations * weights, dim=1).squeeze(0) # [8, 8, 8]
        
        # Apply ReLU to highlight positive contributions
        heatmap = F.relu(weighted_act)
        
        # Interpolate heatmap to original volume shape (64, 64, 64)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0) # [1, 1, 8, 8, 8]
        heatmap = F.interpolate(heatmap, size=(64, 64, 64), mode='trilinear', align_corners=False)
        heatmap = heatmap.squeeze(0).squeeze(0) # [64, 64, 64]
        
        # Convert to numpy and normalize
        heatmap_np = heatmap.detach().cpu().numpy()
        max_val = np.max(heatmap_np)
        if max_val > 0:
            heatmap_np = heatmap_np / max_val
            
        return heatmap_np
