import os
import sys
import subprocess

def check_and_install_dependencies():
    """
    Verifies that all core dependencies are installed, auto-installing any that are missing.
    """
    required_packages = {
        'torch': 'torch',
        'monai': 'monai',
        'streamlit': 'streamlit',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'nibabel': 'nibabel',
        'scipy': 'scipy'
    }
    
    print("=" * 60)
    print("PROJECT #45: INITIALIZING SYSTEM DEPENDENCY VERIFICATION")
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
            except Exception as e:
                print(f"  [WARN] Failed to install {pip_name}: {e}. Proceeding with built-in fallbacks.")
    print("-" * 60)


def generate_mock_weights():
    """
    Pre-compiles dummy weights from model.py to ensure the model branch
    initializes instantly without runtime failures or missing checkpoints.
    """
    weights_path = os.path.join(os.path.dirname(__file__), "weights.pth")
    if os.path.exists(weights_path):
        print("  [OK] Model checkpoint weights found.")
        return
        
    print("  [..] Generating dummy weights for MultimodalLungNet...")
    try:
        import torch
        from model import MultimodalLungNet
        
        # Instantiate model and save state dictionary
        model = MultimodalLungNet()
        torch.save(model.state_dict(), weights_path)
        print("  [OK] weights.pth successfully created!")
    except Exception as e:
        print(f"  [WARNING] Could not pre-generate weights: {e}")
    print("-" * 60)


def main():
    check_and_install_dependencies()
    generate_mock_weights()
    
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    print(f"\n[LAUNCHING] Starting Streamlit dashboard on {app_path}...\n")
    
    try:
        # Launch streamlit dashboard
        # Using sys.executable to ensure we execute streamlit in the exact same environment
        # Adding --server.headless=true suppresses interactive email prompts
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"])
    except KeyboardInterrupt:
        print("\n[STOPPED] Dashboard terminated by user.")
    except Exception as e:
        print(f"[FATAL] Streamlit failed to launch: {e}")


if __name__ == "__main__":
    main()
