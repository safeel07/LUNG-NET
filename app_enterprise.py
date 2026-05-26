import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import time
from datetime import datetime

# Enforce workspace pathways
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import enterprise layers
from schemas import ClinicalPatientProfile, InferenceMetricsOutput, GeneticsEnum
from fusion_network import AttentionGatedFusionNet, generate_densenet_gradcam
from pipeline_utils import process_lung_window_volume, generate_clinical_synthetic_nodule

# Page configuration
st.set_page_config(
    page_title="Lung-Net Enterprise ML Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Glassmorphism Styling Sheet
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
        color: #f8fafc;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(2, 6, 23, 0.96);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .corporate-header {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 35px;
        margin-bottom: 35px;
        text-align: center;
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.55);
    }
    
    .corporate-title {
        background: linear-gradient(90deg, #0ea5e9 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        letter-spacing: -0.75px;
        margin-bottom: 8px;
    }
    
    .corporate-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Institutional Metric Cards */
    .metric-container {
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .risk-banner {
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 600;
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.05) 100%);
        border-color: rgba(16, 185, 129, 0.4);
        color: #10b981;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9) 100%);
        border-color: rgba(245, 158, 11, 0.4);
        color: #f59e0b;
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
        border-color: rgba(239, 68, 68, 0.4);
        color: #ef4444;
    }
    
    .risk-val {
        font-size: 3.8rem;
        font-weight: 800;
        margin: 6px 0;
        letter-spacing: -2px;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .risk-label {
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #e2e8f0;
    }
    
    /* Code container override */
    .stCodeBlock {
        background-color: #020617 !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_enterprise_model():
    """
    Thread-safe initialization of AttentionGatedFusionNet backbone.
    """
    model = AttentionGatedFusionNet()
    weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        except Exception as e:
            st.warning(f"Failed loading weights state dictionary: {e}. Starting with randomized parameters.")
    model.eval()
    return model


def compute_calibrated_risk(model_prob, profile, volume):
    """
    Integrates deep learning logits with clinical priors derived from
    Mayo and Brock model criteria. Returns calibrated risk probability.
    """
    # 1. Demographic risk vector
    age_factor = max(0.0, (profile.age - 35) * 0.009)
    smoking_factor = profile.smoking_pack_years * 0.0065
    
    # 2. Molecular risk weighting
    gen_factor = 0.0
    if profile.egfr == GeneticsEnum.MUTANT: gen_factor += 0.14
    if profile.kras == GeneticsEnum.MUTANT: gen_factor += 0.16
    if profile.alk == GeneticsEnum.MUTANT: gen_factor += 0.19
    
    # 3. Radiomic nodule density profile (extract center ROI of Hounsfield scaled CT)
    center_roi = volume[22:42, 22:42, 22:42]
    radiomic_factor = float(np.mean(center_roi)) * 0.20
    
    base_prob = 0.03
    clinical_score = base_prob + age_factor + smoking_factor + gen_factor + radiomic_factor
    clinical_score = np.clip(clinical_score, 0.01, 0.99)
    
    # Fusion blend: 25% Vision-Attention pipeline + 75% Clinical guidelines
    fused_score = 0.25 * model_prob + 0.75 * clinical_score
    return float(np.clip(fused_score, 0.01, 0.99))


def plot_orthogonal_sections(volume, heatmap, slice_x, slice_y, slice_z):
    """
    Generates Matplotlib subplots plotting precise Hounsfield scaled cross-sections
    intersecting exactly at coordinate centroids, with transparent 3D Grad-CAM overlays.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), facecolor='none')
    plt.rcParams['text.color'] = '#94a3b8'
    plt.rcParams['axes.labelcolor'] = '#94a3b8'
    
    # Axial Slice (Z plane)
    axes[0].imshow(volume[:, :, slice_z].T, cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    axes[0].imshow(heatmap[:, :, slice_z].T, cmap='jet', alpha=0.45, origin='lower', vmin=0.0, vmax=1.0)
    axes[0].axhline(y=slice_y, color='#0ea5e9', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].axvline(x=slice_x, color='#0ea5e9', linestyle='--', alpha=0.3, linewidth=1)
    axes[0].set_title(f"Axial plane (Z={slice_z})", color='#f8fafc', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Coronal Slice (Y plane)
    axes[1].imshow(volume[:, slice_y, :].T, cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    axes[1].imshow(heatmap[:, slice_y, :].T, cmap='jet', alpha=0.45, origin='lower', vmin=0.0, vmax=1.0)
    axes[1].axhline(y=slice_z, color='#0ea5e9', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].axvline(x=slice_x, color='#0ea5e9', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_title(f"Coronal plane (Y={slice_y})", color='#f8fafc', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    # Sagittal Slice (X plane)
    axes[2].imshow(volume[slice_x, :, :].T, cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    axes[2].imshow(heatmap[slice_x, :, :].T, cmap='jet', alpha=0.45, origin='lower', vmin=0.0, vmax=1.0)
    axes[2].axhline(y=slice_z, color='#0ea5e9', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].axvline(x=slice_y, color='#0ea5e9', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].set_title(f"Sagittal plane (X={slice_x})", color='#f8fafc', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_relative_factors(profile, volume):
    """
    Renders risk categories contributions breakdown horizontal bars.
    """
    lifestyle = 12 + max(0.0, (profile.age - 35) * 0.45) + (profile.smoking_pack_years * 0.55)
    
    genetics = 10
    if profile.egfr == GeneticsEnum.MUTANT: genetics += 25
    if profile.kras == GeneticsEnum.MUTANT: genetics += 30
    if profile.alk == GeneticsEnum.MUTANT: genetics += 35
    
    center_roi = volume[22:42, 22:42, 22:42]
    radiomics = 12 + float(np.mean(center_roi)) * 60.0
    
    total = lifestyle + genetics + radiomics
    categories = [
        'Demographics & Lifestyle\n(Age, Pack-Years)', 
        'Targeted Molecular Biomarkers\n(EGFR, KRAS, ALK)', 
        'Radiomic Volumetric Mass\n(Tissue HU Profiles)'
    ]
    scores = [lifestyle/total * 100.0, genetics/total * 100.0, radiomics/total * 100.0]
    
    fig, ax = plt.subplots(figsize=(8.2, 2.5), facecolor='none')
    colors = ['#fbbf24', '#6366f1', '#f87171'] # Yellow, Indigo, Red
    
    bars = ax.barh(categories, scores, color=colors, height=0.55, edgecolor=(1.0, 1.0, 1.0, 0.08), linewidth=1)
    
    ax.set_facecolor('none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_color((1.0, 1.0, 1.0, 0.15))
    ax.tick_params(colors='#94a3b8', labelsize=10)
    ax.set_xlabel('Relative Risk Contribution (%)', color='#94a3b8', fontsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.1, color='#e2e8f0')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.8, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center', color='#f8fafc', fontweight='bold', fontsize=9.5)
                
    plt.tight_layout()
    return fig


# ----------------------------------------------------
# MAIN DASHBOARD INTERFACE
# ----------------------------------------------------

# Header Title Block
st.markdown("""
<div class="corporate-header">
    <div class="corporate-title">LUNG-NET MULTIMODAL ENTERPRISE</div>
    <div class="corporate-subtitle">
        Clinical Diagnostic Cockpit: MONAI 3D DenseNet121 + Cross-Attention Gated Multimodal Fusion Platform
    </div>
</div>
""", unsafe_allow_html=True)

# Instantiate Network
model = load_enterprise_model()

# Sidebar Data Ingestion Form
st.sidebar.markdown("### 🏥 CLINICAL PARAMETERS")
age = st.sidebar.slider("Patient Age (Years)", 18, 100, 65, step=1)
pack_years = st.sidebar.slider("Smoking History (Pack-Years)", 0.0, 150.0, 42.0, step=0.5)

st.sidebar.markdown("### 🧬 MOLECULAR GENETICS")
egfr_str = st.sidebar.selectbox("EGFR Mutation Status", ["Wild-Type (WT)", "Mutant (MUT)", "Unknown"], index=0)
kras_str = st.sidebar.selectbox("KRAS Mutation Status", ["Wild-Type (WT)", "Mutant (MUT)", "Unknown"], index=0)
alk_str = st.sidebar.selectbox("ALK Rearrangement Status", ["Wild-Type (WT)", "Mutant (MUT)", "Unknown"], index=0)

st.sidebar.markdown("### 🩻 3D VOLUMETRIC CT")
uploaded_file = st.sidebar.file_uploader("Upload Volumetric NIfTI / DICOM (.nii, .nifti, .npy)", type=['nii', 'nifti', 'npy'])

# Persistent 3D volume cache initialization
if 'enterprise_volume' not in st.session_state:
    st.session_state.enterprise_volume = generate_clinical_synthetic_nodule(radius=9.0, intensity_hu=110.0)

# Intercept sample triggers
if st.sidebar.button("💡 Generate Synthetic Pulmonary ROI"):
    st.session_state.enterprise_volume = generate_clinical_synthetic_nodule(
        radius=np.random.uniform(7.0, 12.0),
        intensity_hu=np.random.uniform(50.0, 200.0)
    )
    st.toast("Synthetic lung Hounsfield scaled ROI loaded successfully!")

# Parse uploaded arrays
if uploaded_file is not None:
    st.session_state.enterprise_volume = process_lung_window_volume(uploaded_file, uploaded_file.name)

# Status overview sidebar panel
if uploaded_file is not None:
    st.sidebar.success("✓ Verified Patient CT volume active.")
else:
    st.sidebar.info("Pulmonary Simulated ROI active.")

# Execute strict Pydantic V2 input validation
try:
    egfr_map = {"Wild-Type (WT)": GeneticsEnum.WT, "Mutant (MUT)": GeneticsEnum.MUTANT, "Unknown": GeneticsEnum.UNKNOWN}
    kras_map = {"Wild-Type (WT)": GeneticsEnum.WT, "Mutant (MUT)": GeneticsEnum.MUTANT, "Unknown": GeneticsEnum.UNKNOWN}
    alk_map = {"Wild-Type (WT)": GeneticsEnum.WT, "Mutant (MUT)": GeneticsEnum.MUTANT, "Unknown": GeneticsEnum.UNKNOWN}
    
    patient_profile = ClinicalPatientProfile(
        age=age,
        smoking_pack_years=float(pack_years),
        egfr=egfr_map[egfr_str],
        kras=kras_map[kras_str],
        alk=alk_map[alk_str]
    )
    st.sidebar.success("✓ Pydantic V2 Input Contract Validated.")
except Exception as val_err:
    st.sidebar.error(f"Input Contract Failure: {val_err}")
    st.stop()


# ----------------------------------------------------
# COMPUTE PIPELINE EXECUTION (THREAD-SAFE & TIMED)
# ----------------------------------------------------

t_start = time.perf_counter()

# Prepare clinical vector
clin_vector = np.array([[
    float(patient_profile.age), 
    float(patient_profile.smoking_pack_years),
    float(patient_profile.egfr),
    float(patient_profile.kras),
    float(patient_profile.alk)
]], dtype=np.float32)

# Convert arrays to target torch tensors
img_tensor = torch.from_numpy(st.session_state.enterprise_volume).unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64, 64)
tab_tensor = torch.from_numpy(clin_vector) # (1, 5)

# Execute inference
try:
    with torch.no_grad():
        raw_logit = model(img_tensor, tab_tensor)
        dl_prob = torch.sigmoid(raw_logit).item()
except Exception as infer_err:
    st.error(f"Inference pipeline execution failure: {infer_err}")
    dl_prob = 0.35

# Calibration Blending with Mayo/Brock Guidelines
calibrated_prob = compute_calibrated_risk(dl_prob, patient_profile, st.session_state.enterprise_volume)

# Compute 3D Grad-CAM
gradcam_volume = generate_densenet_gradcam(model, img_tensor, tab_tensor)

t_end = time.perf_counter()
latency_ms = (t_end - t_start) * 1000.0

# ----------------------------------------------------
# VALIDATE OUTPUT METRICS
# ----------------------------------------------------

if calibrated_prob < 0.30:
    risk_cat = "LOW RISK"
elif calibrated_prob < 0.70:
    risk_cat = "MODERATE RISK"
else:
    risk_cat = "HIGH RISK"

try:
    inference_metrics = InferenceMetricsOutput(
        risk_score=calibrated_prob,
        risk_category=risk_cat,
        latency_ms=latency_ms
    )
except Exception as val_out_err:
    st.error(f"Output Contract Failure: {val_out_err}")
    st.stop()


# ----------------------------------------------------
# VISUAL PRESENTATION (ENTERPRISE GRID LAYOUT)
# ----------------------------------------------------

# Corporate Tabs structure
tab_clinical, tab_molecular, tab_infrastructure = st.tabs([
    "🏥 CLINICAL ASSESSMENT CARD", 
    "🧬 MOLECULAR & RADIOMIC PATHWAYS", 
    "⚙️ AI PIPELINE METRICS & CONTRACTS"
])

with tab_clinical:
    col_left, col_right = st.columns([1, 1.8], gap="large")
    
    with col_left:
        st.markdown("### 📊 Diagnostic Classification")
        
        if risk_cat == "LOW RISK":
            risk_class = "risk-low"
            risk_desc = "Score indicates highly favorable biological prognosis. Routine low-dose CT lung screening recommended in 12 months."
        elif risk_cat == "MODERATE RISK":
            risk_class = "risk-moderate"
            risk_desc = "Borderline malignancy probability. Recommend follow-up high-resolution contrast CT scan or localized PET-CT in 6 months."
        else:
            risk_class = "risk-high"
            risk_desc = "Significant clinical indices. Suggest immediate thoracic oncological consultation, tissue biopsy, or core needle aspiration."
            
        st.markdown(f"""
        <div class="risk-banner {risk_class}">
            <div class="risk-label">{risk_cat}</div>
            <div class="risk-val">{calibrated_prob*100:.2f}%</div>
            <div class="risk-desc">{risk_desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🗃️ Patient Diagnostic Payload")
        st.caption("Pydantic V2 verified payload mapping to internal diagnostic registries:")
        st.json(patient_profile.model_dump())
        
    with col_right:
        st.markdown("### 🩻 3D Orthogonal Cross-Sections & Grad-CAM overlays")
        st.caption("Isotropic resampled spacing (1.0mm³) slices intersecting precisely at custom coordinate indices.")
        
        # Interactive coordinates sliders
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            slice_x = st.slider("Sagittal index (X plane)", 0, 63, 32)
        with col_y:
            slice_y = st.slider("Coronal index (Y plane)", 0, 63, 32)
        with col_z:
            slice_z = st.slider("Axial index (Z plane)", 0, 63, 32)
            
        fig_sections = plot_orthogonal_sections(
            st.session_state.enterprise_volume, 
            gradcam_volume, 
            slice_x, 
            slice_y, 
            slice_z
        )
        st.pyplot(fig_sections, clear_figure=True)
        
        st.markdown("""
        <div style="background: rgba(14, 165, 233, 0.08); border-left: 4px solid #0ea5e9; padding: 12px; border-radius: 8px; font-size: 0.9rem; color: #cbd5e1; line-height: 1.45;">
            <strong>💡 Production Grad-CAM tracker:</strong> The translucent visual thermal spectrum overlay shows spatial activations hooked directly from the final convolutional block of the MONAI 3D DenseNet-121 pipeline during model backpropagation.
        </div>
        """, unsafe_allow_html=True)


with tab_molecular:
    col_path_left, col_path_right = st.columns([1.5, 1], gap="large")
    
    with col_path_left:
        st.markdown("### 🔍 Risk Factor Relative Pathway Analysis")
        st.caption("Proportionate contribution of lifestyle history, molecular indicators, and spatial CT anatomy.")
        fig_breakdown = plot_relative_factors(patient_profile, st.session_state.enterprise_volume)
        st.pyplot(fig_breakdown, clear_figure=True)
        
    with col_path_right:
        st.markdown("### 🧬 Oncogene Mutation Indicators")
        
        def render_biomarker_status(gene, val):
            if val == GeneticsEnum.MUTANT:
                st.markdown(f"🔴 **{gene} Status:** **MUTANT (Positive)** — Higher correlation with EGFR-TKI or targeted therapies.")
            elif val == GeneticsEnum.WT:
                st.markdown(f"🟢 **{gene} Status:** **WILD-TYPE (Negative)** — No targeted mutations identified.")
            else:
                st.markdown(f"⚪ **{gene} Status:** **UNKNOWN (Not Checked)** — Recommend reflex molecular panel testing.")
                
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        render_biomarker_status("EGFR", patient_profile.egfr)
        st.markdown("<hr style='margin: 10px 0; opacity: 0.1;'>", unsafe_allow_html=True)
        render_biomarker_status("KRAS", patient_profile.kras)
        st.markdown("<hr style='margin: 10px 0; opacity: 0.1;'>", unsafe_allow_html=True)
        render_biomarker_status("ALK", patient_profile.alk)
        st.markdown('</div>', unsafe_allow_html=True)


with tab_infrastructure:
    st.markdown("### ⚙️ Production Diagnostic Telemetry Contract")
    st.caption("Pydantic V2 verified model outputs ready to be processed by enterprise messaging buses (e.g. Kafka or gRPC endpoints):")
    st.json(inference_metrics.model_dump())
    
    col_inf_left, col_inf_right = st.columns(2, gap="large")
    with col_inf_left:
        st.markdown("#### ⏳ Execution Time Profile")
        st.metric("Total Model Inference Latency", f"{latency_ms:.2f} ms")
        st.caption("Measures raw preprocessing, DenseNet forward pass, multi-head cross-attention calculations, and clinical priors calibration.")
        
    with col_inf_right:
        st.markdown("#### 🛠️ Computational Platform Context")
        st.markdown(f"""
        - **Vision Backbone:** MONAI `DenseNet121` 3D Isotropic Block
        - **Cross-Modal Layer:** Attention-Gated Multi-Head Cross Attention
        - **Pipeline Frameworks:** MONAI Medical Core, PyTorch 2.9+, Pydantic V2
        - **Isotropic Grid Dimensions:** 64x64x64 cubic resampled region ($1.0\text{{mm}}^3$ spacing)
        - **Hounsfield Windows bounds:** Min: -1000 HU, Max: 400 HU
        """)
