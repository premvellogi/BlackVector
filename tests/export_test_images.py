"""Convert sample patches to browser-friendly PNG images for UI testing."""
import numpy as np
from pathlib import Path
from PIL import Image

CACHE = Path(r"d:\Netra\data\imagery_cache")
OUT = Path(r"d:\Netra\outputs\test_images")
OUT.mkdir(parents=True, exist_ok=True)

# Pick diverse patches with known ground truth
samples = {
    "forest_mixed": "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_55_90",
    "urban_industrial": "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_80_59",
    "agriculture_pasture": "S2A_MSIL2A_20170617T113321_N9999_R080_T29UPU_07_64",
    "water_coastal": "S2A_MSIL2A_20170813T112121_N9999_R037_T29SNC_02_13",
    "dense_vegetation": "S2A_MSIL2A_20170701T093031_N9999_R136_T35VPK_13_80",
}

for label, patch_id in samples.items():
    npz_path = CACHE / f"{patch_id}.npz"
    if not npz_path.exists():
        print(f"  [SKIP] {label}: {npz_path.name} not found")
        continue
    
    data = np.load(npz_path)
    
    # --- 1. OPTICAL (Sentinel-2) ---
    s2 = data["s2"]  # shape: (10, H, W)
    rgb = np.stack([s2[2], s2[1], s2[0]], axis=-1)  # R=B04, G=B03, B=B02
    p2, p98 = np.percentile(rgb, [2, 98])
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    img_opt = Image.fromarray(rgb).resize((512, 512), Image.LANCZOS)
    opt_path = OUT / f"{label}_optical.png"
    img_opt.save(opt_path, "PNG")
    
    # --- 2. SAR RADAR (Sentinel-1) ---
    s1 = data["s1"]  # shape: (2, H, W) -> [VV, VH]
    vv = s1[0]
    vh = s1[1]
    ratio = vv / (vh + 1e-6)
    sar_rgb = np.stack([vv, vh, ratio], axis=-1)
    sp2, sp98 = np.percentile(sar_rgb, [2, 98])
    sar_rgb = np.clip((sar_rgb - sp2) / (sp98 - sp2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    img_sar = Image.fromarray(sar_rgb).resize((512, 512), Image.LANCZOS)
    sar_path = OUT / f"{label}_sar.png"
    img_sar.save(sar_path, "PNG")
    
    print(f"  [OK] {label}:")
    print(f"       Optical -> {opt_path.name}")
    print(f"       SAR     -> {sar_path.name}")

print(f"\nAll test images saved to: {OUT}")
