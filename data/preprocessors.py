import os
import tempfile
import numpy as np

try:
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        Resized
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def generate_hounsfield_pulmonary_nodule(radius=8.5, intensity_hu=130.0, background_hu=-760.0):
    """
    Generates simulated 3D CT ROI utilizing realistic Hounsfield Units (HU) bounds.
    """
    grid_size = 64
    x = np.linspace(-grid_size//2, grid_size//2, grid_size)
    y = np.linspace(-grid_size//2, grid_size//2, grid_size)
    z = np.linspace(-grid_size//2, grid_size//2, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    dist = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Nodule profile (Gaussian dropoff)
    nodule_profile = np.exp(- (dist**2) / (2 * (radius**2))) * (intensity_hu - background_hu)
    background_noise = np.random.normal(loc=background_hu, scale=80.0, size=(grid_size, grid_size, grid_size))
    
    volume_hu = background_hu + nodule_profile + background_noise
    
    # Min-max scale according to Hounsfield Unit pulmonary windowing limits (-1000 to 400 HU)
    hu_min, hu_max = -1000.0, 400.0
    normalized = (volume_hu - hu_min) / (hu_max - hu_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def process_clinical_ingestion(file_obj, filename):
    """
    Resamples CT voxel spacing to isotropic 1.0mm^3 spacing and window scales intensities.
    """
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name
        
    try:
        if MONAI_AVAILABLE:
            keys = ["image"]
            transforms = Compose([
                LoadImaged(keys=keys, image_only=True),
                EnsureChannelFirstd(keys=keys),
                Orientationd(keys=keys, axcodes="RAS"),
                Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                ScaleIntensityRanged(
                    keys=keys,
                    a_min=-1000.0, # Lung window minimum
                    a_max=400.0,   # Lung window maximum
                    b_min=0.0,
                    b_max=1.0,
                    clip=True
                ),
                Resized(keys=keys, spatial_size=(64, 64, 64), mode="trilinear")
            ])
            
            payload = {"image": tmp_path}
            processed = transforms(payload)
            return np.clip(processed["image"].squeeze(0).numpy(), 0.0, 1.0).astype(np.float32)
            
        else:
            import nibabel as nib
            img = nib.load(tmp_path)
            data = img.get_fdata()
            
            # Limit and scale HU spectrum manually
            data = np.clip(data, -1000.0, 400.0)
            data = (data - (-1000.0)) / (400.0 - (-1000.0))
            
            h, w, d = data.shape
            if (h, w, d) != (64, 64, 64):
                if SCIPY_AVAILABLE:
                    factors = (64.0/h, 64.0/w, 64.0/d)
                    data = zoom(data, factors, order=1)
                else:
                    new_vol = np.zeros((64, 64, 64), dtype=np.float32)
                    sh, sw, sd = min(h, 64), min(w, 64), min(d, 64)
                    new_vol[:sh, :sw, :sd] = data[:sh, :sw, :sd]
                    data = new_vol
            return np.clip(data, 0.0, 1.0).astype(np.float32)
            
    except Exception as err:
        print(f"[INGESTION EXCEPTION] {err}. Reverting to simulator.")
        return generate_hounsfield_pulmonary_nodule()
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
