# LUNG-NET: Unified Multimodal Lung Cancer Risk Stratification Platform

LUNG-NET is an **FDA-compliant, institutional-grade clinical AI diagnostic suite** designed to stratify lung cancer malignancy risk by fusing 3D pulmonary CT scan volumes with discrete genomic biomarkers (EGFR, KRAS, ALK mutations) and patient clinical exposures (Age, Smoking Pack-Years).

The platform features two state-of-the-art diagnostic backbones unified under a single clinical visual cockpit client:
1. **3D Swin-Transformer backbone** utilizing multi-stage shifted-window self-attention.
2. **3D DenseNet-121 backbone** utilizing gated convolutional feature average pooling.

Both architectures explicitly model inter-modality dependencies by replacing basic concatenation with a **Multi-Head Cross-Attention Gating layer** (Tabular Query attending to 3D Vision Keys/Values).

---

##  Package Structure & Architecture

The workspace is organized into clean, modular packages:

```
medical-proj/
├── core/
│   ├── __init__.py
│   ├── cnn_fusion_net.py     # MONAI 3D DenseNet121 + Cross-Attention Gating
│   └── swin_fusion_net.py    # 3D Shifted-Window Swin-Transformer + Cross-Attention
├── data/
│   ├── __init__.py
│   ├── schemas.py            # Strict Pydantic V2 demographics & telemetry validators
│   └── preprocessors.py      # MONAI Spacing (1.0mm³) & Hounsfield windows (-1000 to 400 HU)
├── ui/
│   ├── __init__.py
│   └── dashboard.py          # Unified visual cockpit containing Plotly 3D & Matplotlib
├── .gitignore                # Exclude temporary weights & cache
├── README.md                 # Technical guide & manual
└── run.py                    # Master self-healing orchestrator bootstrap launcher
```

---

##  Core System Specifications

### 1. 3D Vision Backbones
*   **Swin-Transformer 3D:** Processes cubic tensors of shape `(B, 1, 64, 64, 64)` through volumetric patch embedding (4x4x4 patches), multi-stage shifted window self-attentions, and downsampling patch merging blocks to harvest `(B, 768)` multi-scale sequence tokens.
*   **MONAI DenseNet-121 3D:** Extracts visual features from cubic ROIs, utilizing adaptive 3D pooling to yield deterministic `(B, 1024)` pooled representation vectors.

### 2. Tabular Co-Embedding Streams
Discrete genetic alteration enums (EGFR, KRAS, ALK mutations) are mapped through independent PyTorch `nn.Embedding(3, 16)` blocks before merging with demographics projections (Age, Pack-Years), mapping clinical patient susceptibility to a continuous `(B, 256)` token.

### 3. Scaled Multi-Head Cross-Attention Gating
Rather than simple vector stitching (`torch.cat`), clinical queries ($Q$) actively attend to visual anatomies ($K$, $V$) to capture gating weights dynamically:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
This models spatial susceptibility gating (e.g. weighting specific nodule regions based on genetic susceptibility) before MLP classification.

### 4. Interactive Volumetric 3D Raycasting XAI
Calculates deterministic **3D Grad-CAM** activation maps by backpropagating logits through Swin self-attentions or DenseNet convolutional layers. The visual cockpit renders an interactive Plotly 3D Volumetric Raycast (`plotly.graph_objects.Volume`), overlaying structural gray-scale CT anatomy with thermal Grad-CAM activations mapped directly to rotatable, zoomable 3D coordinate grids.

---

##  Zero-Setup Quickstart

The master orchestrator `run.py` is equipped with self-healing script hooks. It automatically checks and installs missing dependencies (such as `monai`, `plotly`, `streamlit`), compiles compatible model parameters `weights_cnn.pth` and `weights_swin.pth`, and starts the visual cockpit on port 8501.

To boot the unified platform, simply execute:
```bash
py run.py
```

Open your local browser to navigate the clinical cockpit:
*   **URL:** [http://localhost:8501](http://localhost:8501)
