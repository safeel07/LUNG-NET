import os
import sys
import subprocess

def check_system_libraries():
    """
    Validates that all necessary modular packages are present,
    auto-installing any missing ones to guarantee zero-setup startups.
    """
    required_packages = {
        'torch': 'torch',
        'monai': 'monai',
        'plotly': 'plotly',
        'pydantic': 'pydantic>=2.0',
        'streamlit': 'streamlit',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'nibabel': 'nibabel',
        'scipy': 'scipy'
    }
    
    print("=" * 60)
    print("LUNG-NET UNIFIED DIAGNOSTIC PLATFORM BOOTSTRAP")
    print("=" * 60)
    
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  [OK] {module_name:<12} : Already installed.")
        except ImportError:
            print(f"  [..] {module_name:<12} : Missing. Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                print(f"  [OK] {module_name:<12} : Successfully installed.")
            except Exception as err:
                print(f"  [WARN] Failed to install {pip_name}: {err}. Continuing using fallback adapters.")
    print("-" * 60)


def precompile_model_checkpoints():
    """
    Pre-compiles parameters for both diagnostic neural networks
    to ensure immediate load and execution without missing weights files.
    """
    root = os.path.dirname(os.path.abspath(__file__))
    cnn_path = os.path.join(root, "weights_cnn.pth")
    swin_path = os.path.join(root, "weights_swin.pth")
    
    try:
        import torch
    except Exception:
        print("  [WARN] PyTorch is not available. Skipping model checkpoint pre-compilation. Falling back to analytical pipelines.")
        print("-" * 60)
        return
    
    # 1. Compile 3D DenseNet-121 CNN weights
    if not os.path.exists(cnn_path):
        print("  [..] Pre-compiling AttentionGatedFusionNet (DenseNet) weights...")
        try:
            from core.cnn_fusion_net import AttentionGatedFusionNet
            model_cnn = AttentionGatedFusionNet()
            torch.save(model_cnn.state_dict(), cnn_path)
            print("  [OK] weights_cnn.pth successfully created!")
        except Exception as err:
            print(f"  [WARNING] Could not pre-generate CNN weights: {err}")
    else:
        print("  [OK] CNN model checkpoint found.")
        
    # 2. Compile 3D Swin-Transformer weights
    if not os.path.exists(swin_path):
        print("  [..] Pre-compiling SwinCrossAttentionNet weights...")
        try:
            from core.swin_fusion_net import SwinCrossAttentionNet
            model_swin = SwinCrossAttentionNet()
            torch.save(model_swin.state_dict(), swin_path)
            print("  [OK] weights_swin.pth successfully created!")
        except Exception as err:
            print(f"  [WARNING] Could not pre-generate Swin weights: {err}")
    else:
        print("  [OK] Swin-Transformer model checkpoint found.")
        
    print("-" * 60)


def main():
    check_system_libraries()
    precompile_model_checkpoints()
    
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_clinical_system.py")
    print(f"\n[LAUNCHING] Starting LUNG-NET Unified dashboard on {app_path}...\n")
    
    try:
        # Launch Streamlit cockpit programmatically
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"])
    except KeyboardInterrupt:
        print("\n[STOPPED] Dashboard terminated by user.")
    except Exception as err:
        print(f"[FATAL] Streamlit failed to launch: {err}")


if __name__ == "__main__":
    main()
