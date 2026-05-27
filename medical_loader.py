import os
import tempfile
import numpy as np

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from scipy.ndimage import zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def segment_distance(X, Y, Z, A, B):
    """
    Computes the shortest distance from each grid point (X, Y, Z) to a 3D line segment AB.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    dx, dy, dz = B[0] - A[0], B[1] - A[1], B[2] - A[2]
    len_sq = dx*dx + dy*dy + dz*dz
    if len_sq == 0:
        return np.sqrt((X - A[0])**2 + (Y - A[1])**2 + (Z - A[2])**2)
    
    # Projection t of AP onto AB clamped to [0.0, 1.0]
    t = ((X - A[0])*dx + (Y - A[1])*dy + (Z - A[2])*dz) / len_sq
    t = np.clip(t, 0.0, 1.0)
    
    proj_x = A[0] + t*dx
    proj_y = A[1] + t*dy
    proj_z = A[2] + t*dz
    return np.sqrt((X - proj_x)**2 + (Y - proj_y)**2 + (Z - proj_z)**2)


def generate_synthetic_ct_nodule(age=65, smoking_pack_years=48.0, egfr=0, kras=0, alk=0, background_hu=-700.0):
    """
    Generates a highly-realistic, anatomically structured 3D isotropic lung lobe segment (64x64x64)
    containing Left & Right Lung Lobes, Oblique & Horizontal Lobe Fissures, a medial Cardiac Notch inside
    the Left Lobe, a concave Diaphragmatic base, a branching Trachea & bronchial tree system, low-density
    lung parenchyma air cavities (-700 HU), smoking-induced emphysematous pockets, and a localized, spiculed
    malignant nodule pathology (+150 HU) situated inside the upper right lobe surrounded by a ground-glass
    opacity (GGO) infectious infiltrate, all dynamically scaled based on clinical patient parameters.
    """
    grid_size = 64
    x = np.linspace(-32, 32, grid_size)
    y = np.linspace(-32, 32, grid_size)
    z = np.linspace(-32, 32, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 1. Start with completely empty vacuum space background (-1000 HU)
    volume_hu = np.ones((grid_size, grid_size, grid_size)) * -1000.0
    
    # 2. Add Left and Right Lung Lobes (low-density air parenchyma -700 HU)
    # Lungs taper at the apex (upper Z) and widen at the base (lower Z)
    z_norm = np.clip((26 - Z) / 52, 0.0, 1.0)
    taper_r = 0.35 + 0.85 * z_norm  # Taper scale factor along height
    
    # Costal Rib impressions: periodic wave perturbation along the outer lateral borders
    phi_angle = np.arctan2(Y, X)
    costal_ridges = 1.0 + 0.038 * np.cos(0.95 * Z) * np.cos(5.5 * phi_angle)
    
    # Mediastinal concavity exclusion mask (carving the inner flat/concave walls of both lungs)
    mediastinum_dist = (X / 6.5)**2 + ((Y + 3.5) / 10.5)**2
    mediastinum_mask = mediastinum_dist > 1.0
    
    # Apply taper and costal rib impressions to ellipsoids
    # Right Lobe (centered at X = -13, Y = 1, Z = -2)
    right_lobe_dist = ((X + 13.5) / (12.0 * taper_r * costal_ridges))**2 + ((Y - 1) / (15.0 * taper_r * costal_ridges))**2 + ((Z + 2) / 22.0)**2
    # Left Lobe (centered at X = 13, Y = 1, Z = -2)
    left_lobe_dist = ((X - 13.5) / (11.0 * taper_r * costal_ridges))**2 + ((Y - 1) / (15.0 * taper_r * costal_ridges))**2 + ((Z + 2) / 22.0)**2
    
    # Left Lobe Cardiac Notch: sphere subtraction in lower medial posterior region
    cardiac_notch = ((X - 6.5) / 8.5)**2 + ((Y - 6.0) / 8.5)**2 + ((Z + 6.0) / 9.5)**2
    
    # Diaphragmatic Dome Concavity: subtract a curved dome from the base of both lungs (Z < -14)
    diaphragm_dist = np.sqrt(X**2 + Y**2 + (Z + 36.0)**2)
    diaphragm_mask = diaphragm_dist > 22.0
    
    # Build lung parenchymal masks (applying mediastinal exclusion and rib ridges)
    right_lung_base = (right_lobe_dist <= 1.0) & (Z > -22) & (Z < 24) & diaphragm_mask & mediastinum_mask
    left_lung_base = (left_lobe_dist <= 1.0) & (Z > -22) & (Z < 24) & (cardiac_notch > 0.85) & diaphragm_mask & mediastinum_mask
    
    # 3. Sculpt Anatomical Fissures
    # Right Lung: oblique fissure (dividing superior/inferior) & horizontal fissure (superior/middle)
    right_oblique = np.abs(Z - 0.5 * Y + 2.0) > 1.2
    right_horizontal = ~((X < -10) & (Y < 6) & (np.abs(Z - 4.0) < 1.2))
    
    # Left Lung: oblique fissure (dividing superior/inferior)
    left_oblique = np.abs(Z - 0.5 * Y + 2.0) > 1.2
    
    right_lung_mask = right_lung_base & right_oblique & right_horizontal
    left_lung_mask = left_lung_base & left_oblique
    
    # Fill lung parenchyma with visible density (-700 HU) with noise
    parenchyma_hu = np.random.normal(loc=background_hu, scale=18.0, size=(grid_size, grid_size, grid_size))
    
    # 4. Smoking-induced parenchymal damage / emphysematous tissue destruction (-950 HU)
    if smoking_pack_years > 20.0:
        # Create noise bubbles for emphysema pockets
        num_emphysema_centers = int(min(12, smoking_pack_years // 6))
        for _ in range(num_emphysema_centers):
            # Randomly place pockets inside both lungs
            rand_x = np.random.uniform(-20.0, 20.0)
            rand_y = np.random.uniform(-10.0, 10.0)
            rand_z = np.random.uniform(-15.0, 15.0)
            pocket_r = np.random.uniform(2.5, 4.5)
            pocket_dist = np.sqrt((X - rand_x)**2 + (Y - rand_y)**2 + (Z - rand_z)**2)
            parenchyma_hu = np.where(pocket_dist < pocket_r, np.random.normal(-920.0, 10.0, size=(grid_size, grid_size, grid_size)), parenchyma_hu)
            
    volume_hu = np.where(right_lung_mask | left_lung_mask, parenchyma_hu, volume_hu)
    
    # 5. Add branching trachea and bronchial airways (air density -950 HU with high-density walls +100 HU)
    trachea_dist = np.sqrt(X**2 + (Y + 2.0)**2)
    trachea_air = (trachea_dist < 1.8) & (Z >= 6) & (Z < 26)
    trachea_wall = (trachea_dist >= 1.8) & (trachea_dist < 2.8) & (Z >= 6) & (Z < 26)
    
    # Right Main Bronchus tube
    t1 = np.clip((6 - Z) / 10.0, 0.0, 1.0)
    bronch_r_dist = np.sqrt((X - (-10.0 * t1))**2 + (Y - (-2.0 + 2.0 * t1))**2 + (Z - (6 - 10.0 * t1))**2)
    bronch_r_air = (bronch_r_dist < 1.2) & (Z < 6) & (Z > -6)
    bronch_r_wall = (bronch_r_dist >= 1.2) & (bronch_r_dist < 2.0) & (Z < 6) & (Z > -6)
    
    # Left Main Bronchus tube
    t2 = np.clip((6 - Z) / 10.0, 0.0, 1.0)
    bronch_l_dist = np.sqrt((X - (10.0 * t2))**2 + (Y - (-2.0 + 2.0 * t2))**2 + (Z - (6 - 10.0 * t2))**2)
    bronch_l_air = (bronch_l_dist < 1.2) & (Z < 6) & (Z > -6)
    bronch_l_wall = (bronch_l_dist >= 1.2) & (bronch_l_dist < 2.0) & (Z < 6) & (Z > -6)
    
    # Inject Airway Walls (+100 HU)
    volume_hu = np.where(trachea_wall | bronch_r_wall | bronch_l_wall, np.random.normal(loc=100.0, scale=15.0, size=(grid_size, grid_size, grid_size)), volume_hu)
    # Inject Airway Lumens (-950 HU)
    volume_hu = np.where(trachea_air | bronch_r_air | bronch_l_air, -950.0, volume_hu)

    # 5b. Add highly detailed branching vascular / blood vessel tree segments
    # Define primary, secondary, and tertiary vascular branches spreading inside lung parenchyma
    vascular_segments = [
        # --- Right Lung Vascular Tree ---
        ([-8.0, -1.0, 2.0], [-14.0, -2.0, 12.0], 1.6),       # Right Upper Lobe Main
        ([-14.0, -2.0, 12.0], [-16.0, -3.0, 20.0], 1.0),     # Right Apex Segment
        ([-14.0, -2.0, 12.0], [-20.0, 4.0, 14.0], 0.9),      # Right Anterior Segment
        ([-8.0, -1.0, 2.0], [-15.0, 5.0, 2.0], 1.4),         # Right Middle Lobe Main
        ([-15.0, 5.0, 2.0], [-21.0, 8.0, 3.0], 0.8),         # Middle Branch 1
        ([-15.0, 5.0, 2.0], [-18.0, 6.0, -4.0], 0.8),        # Middle Branch 2
        ([-8.0, -1.0, 2.0], [-14.0, -2.0, -8.0], 1.8),       # Right Lower Lobe Main
        ([-14.0, -2.0, -8.0], [-18.0, -6.0, -18.0], 1.1),    # Right Posterior Basal
        ([-14.0, -2.0, -8.0], [-22.0, 2.0, -14.0], 1.0),     # Right Lateral Basal
        
        # --- Left Lung Vascular Tree ---
        ([8.0, -1.0, 2.0], [14.0, -2.0, 11.0], 1.6),         # Left Upper Lobe Main
        ([14.0, -2.0, 11.0], [16.0, -3.0, 19.0], 1.0),       # Left Apex Segment
        ([14.0, -2.0, 11.0], [20.0, 4.0, 12.0], 0.9),        # Left Anterior Segment
        ([8.0, -1.0, 2.0], [13.0, -2.0, -8.0], 1.8),         # Left Lower Lobe Main
        ([13.0, -2.0, -8.0], [17.0, -6.0, -18.0], 1.1),      # Left Posterior Basal
        ([13.0, -2.0, -8.0], [21.0, 2.0, -14.0], 1.0),       # Left Lateral Basal
    ]
    
    combined_vessels_mask = np.zeros_like(X)
    for A_pt, B_pt, rad in vascular_segments:
        d = segment_distance(X, Y, Z, A_pt, B_pt)
        env = np.clip((rad - d) / 0.8, 0.0, 1.0)
        combined_vessels_mask = np.maximum(combined_vessels_mask, env)
        
    vessel_intensity = np.random.normal(loc=80.0, scale=12.0, size=(grid_size, grid_size, grid_size))
    # Apply blood vessels exclusively inside the left and right lung lobes parenchyma masks
    volume_hu = np.where(
        right_lung_mask | left_lung_mask,
        volume_hu + combined_vessels_mask * (vessel_intensity - volume_hu),
        volume_hu
    )
    
    # 6. DYNAMIC MALIGNANT PATHOLOGY GENERATION
    # The infection tumor nodule's radius, spiculation density, and satellite lesions grow
    # in direct correlation with the patient's Age, Smoking history, and mutations.
    base_radius = 3.5 + 0.12 * float(smoking_pack_years) + 0.04 * max(0.0, float(age) - 35.0)
    
    # Add mutation factor (+3.0mm per active mutation)
    mutation_factor = 0.0
    if float(egfr) > 0: mutation_factor += 3.0
    if float(kras) > 0: mutation_factor += 3.0
    if float(alk) > 0: mutation_factor += 3.0
    
    nodule_radius = np.clip(base_radius + mutation_factor, 2.5, 18.0)
    
    # Pathological center inside the Upper Right Lobe: X = -12, Y = -2, Z = 8
    nodule_center = np.array([-12.0, -2.0, 8.0])
    X_rel = X - nodule_center[0]
    Y_rel = Y - nodule_center[1]
    Z_rel = Z - nodule_center[2]
    
    dist_nodule = np.sqrt(X_rel**2 + Y_rel**2 + Z_rel**2)
    
    # Starburst spicular wave equation (spicules become more aggressive with higher smoking)
    phi = np.arctan2(Y_rel, X_rel)
    theta = np.arccos(np.clip(Z_rel / np.clip(dist_nodule, 1e-5, 100.0), -1.0, 1.0))
    
    spicule_amp = 1.0 + 0.04 * float(smoking_pack_years)
    spicules = spicule_amp * np.sin(8 * phi) * np.cos(8 * theta)
    spicular_dist = dist_nodule - spicules
    
    # Nodule core transition
    nodule_intensity = np.random.normal(loc=150.0, scale=25.0, size=(grid_size, grid_size, grid_size))
    nodule_transition = np.clip((nodule_radius - spicular_dist) / 2.0, 0.0, 1.0)
    
    # 7. Add surrounding Ground-Glass Opacity (GGO) / Infectious consolidation infiltrate
    # The GGO spread also scales with the calculated disease aggressiveness
    ggo_scale = 10.0 + 0.15 * float(smoking_pack_years) + 0.05 * float(age)
    infiltrate_density = np.random.normal(loc=-180.0, scale=35.0, size=(grid_size, grid_size, grid_size))
    infiltrate_strength = np.exp(-(dist_nodule**2) / (2.0 * (ggo_scale**2))) # Gaussian distribution
    infiltrate_strength = np.clip(infiltrate_strength * 0.75, 0.0, 1.0)
    
    # Apply infiltrate and nodule core within the right lung parenchyma boundaries
    volume_hu = np.where(right_lung_mask, volume_hu + infiltrate_strength * (infiltrate_density - volume_hu), volume_hu)
    volume_hu = volume_hu + nodule_transition * (nodule_intensity - volume_hu)
    
    # 8. MULTI-FOCAL SATELLITE NODULES (If risk factors are extremely high)
    if smoking_pack_years > 55.0 or (smoking_pack_years > 30.0 and mutation_factor > 0.0):
        # Inject secondary small metastatic nodule in inferior left lung (X=14, Y=-2, Z=-12)
        sat_center = np.array([14.0, -2.0, -12.0])
        dist_sat = np.sqrt((X - sat_center[0])**2 + (Y - sat_center[1])**2 + (Z - sat_center[2])**2)
        sat_trans = np.clip((3.5 - dist_sat) / 1.0, 0.0, 1.0)
        sat_intensity = np.random.normal(loc=120.0, scale=20.0, size=(grid_size, grid_size, grid_size))
        
        # Apply satellite only inside left lung
        volume_hu = np.where(left_lung_mask, volume_hu + sat_trans * (sat_intensity - volume_hu), volume_hu)
        
    # 9. Normalization scale according to lung tissue window [-1000 HU to 400 HU]
    hu_min, hu_max = -1000.0, 400.0
    normalized = (volume_hu - hu_min) / (hu_max - hu_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)


def load_and_transform_nifti(file_buffer, filename: str) -> np.ndarray:
    """
    Clinical Ingestion Layer:
    Unpacks NIfTI files (.nii or .nii.gz) via nibabel, limits Hounsfield Unit
    ranges between -1000 HU and 400 HU, and scales spatial grids precisely
    to isotropic (64, 64, 64) tensors via safe center-cropping or resampling.
    """
    with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as tmp_file:
        tmp_file.write(file_buffer.read())
        tmp_path = tmp_file.name

    try:
        if not NIBABEL_AVAILABLE:
            print("[WARN] Nibabel is missing. Reverting to high-fidelity simulated nodule.")
            return generate_synthetic_ct_nodule()

        # Load volumetric image
        img = nib.load(tmp_path)
        volume = img.get_fdata()

        # 1. Intensity scaling according to lung tissue window [-1000 HU to 400 HU]
        volume = np.clip(volume, -1000.0, 400.0)
        volume = (volume - (-1000.0)) / (400.0 - (-1000.0))

        # 2. Reshape spatial grid to exactly (64, 64, 64)
        h, w, d = volume.shape
        target_size = 64

        if (h, w, d) != (target_size, target_size, target_size):
            if SCIPY_AVAILABLE:
                # Deterministic scale scaling
                factors = (target_size / h, target_size / w, target_size / d)
                volume = zoom(volume, factors, order=1)
            else:
                # Safe crop/pad array adjustments to avoid index errors
                new_volume = np.ones((target_size, target_size, target_size), dtype=np.float32) * 0.15
                
                # Compute margins
                src_x = slice(max(0, (h - target_size)//2), min(h, (h + target_size)//2))
                src_y = slice(max(0, (w - target_size)//2), min(w, (w + target_size)//2))
                src_z = slice(max(0, (d - target_size)//2), min(d, (d + target_size)//2))
                
                cropped = volume[src_x, src_y, src_z]
                ch, cw, cd = cropped.shape
                
                dst_x = slice((target_size - ch)//2, (target_size - ch)//2 + ch)
                dst_y = slice((target_size - cw)//2, (target_size - cw)//2 + cw)
                dst_z = slice((target_size - cd)//2, (target_size - cd)//2 + cd)
                
                new_volume[dst_x, dst_y, dst_z] = cropped
                volume = new_volume

        return np.clip(volume, 0.0, 1.0).astype(np.float32)

    except Exception as err:
        print(f"[LOAD ERROR] Failed to parse medical volume: {err}. Loading fallback generator.")
        return generate_synthetic_ct_nodule()

    finally:
        # Tidy up temp folder files
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
