#!/usr/bin/env python3
"""
seg.py — Generate tumor segmentation maps for BraTSReg dataset
using DeepMedic inference.
"""

import os
import glob
import subprocess
import shutil

# ===============================================================
# CONFIG
# ===============================================================
DATAPATH = "/workspace/DIRAC/Data/BraTSReg/BraTSReg_Training_Data_v2"
OUTPUT_SUFFIX = "_0000_tumorseg.nii.gz"
OUTPUT_DIR = os.path.join(DATAPATH)

DEEPMEDIC_BIN = "deepMedicRun"  # DeepMedic executable
MODEL_CFG = "/workspace/DIRAC/Model/deepmedic_tumor_model/modelConfig.cfg"  # trained model

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================================================
# SEGMENTATION FUNCTION
# ===============================================================
def run_deepmedic(case_dir):
    case_name = os.path.basename(case_dir)
    out_case_dir = os.path.join(OUTPUT_DIR, case_name)
    os.makedirs(out_case_dir, exist_ok=True)

    # Build test list file for DeepMedic
    test_list_path = os.path.join(out_case_dir, "testList.txt")
    t1 = glob.glob(os.path.join(case_dir, "*_0000_t1.nii.gz"))[0]
    t1ce = glob.glob(os.path.join(case_dir, "*_0000_t1ce.nii.gz"))[0]
    t2 = glob.glob(os.path.join(case_dir, "*_0000_t2.nii.gz"))[0]
    flair = glob.glob(os.path.join(case_dir, "*_0000_flair.nii.gz"))[0]

    with open(test_list_path, "w") as f:
        f.write(f"{t1}\n{t1ce}\n{t2}\n{flair}\n")

    # Run DeepMedic inference
    cmd = [
        DEEPMEDIC_BIN,
        "-model", MODEL_CFG,
        "-test", test_list_path
    ]
    print(f"[RUN] DeepMedic for {case_name}...")
    subprocess.run(cmd, check=True)

    # DeepMedic outputs *_segmentation.nii.gz or *_probabilities.nii.gz
    pred_files = glob.glob(os.path.join(out_case_dir, "*_segmentation.nii.gz"))
    if not pred_files:
        pred_files = glob.glob(os.path.join(out_case_dir, "*_probabilities.nii.gz"))

    if pred_files:
        seg_out_path = os.path.join(case_dir, case_name + OUTPUT_SUFFIX)
        shutil.move(pred_files[0], seg_out_path)
        print(f"[SAVED] Segmentation saved to {seg_out_path}")
    else:
        print(f"[WARN] No segmentation output found for {case_name}")


# ===============================================================
# MAIN
# ===============================================================
def main():
    case_dirs = sorted(glob.glob(os.path.join(DATAPATH, "BraTSReg_*")))
    print(f"Found {len(case_dirs)} cases to segment.")

    for idx, case_dir in enumerate(case_dirs):
        case_name = os.path.basename(case_dir)
        out_path = os.path.join(case_dir, case_name + OUTPUT_SUFFIX)

        try:
            run_deepmedic(case_dir)
        except Exception as e:
            print(f"[FAIL] {case_name} → {e}")

    print("\n✅ All tumor segmentations complete!")


if __name__ == "__main__":
    main()


