# evaluation_fixed_coords_refined.py
import os
import glob
import time
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.model_selection import KFold
from scipy.ndimage import map_coordinates, binary_dilation

# Model imports
from bratsreg_model_stage import (
    UPIR_LDR_laplacian_unit_disp_add_AdaIn_lvl1,
    UPIR_LDR_laplacian_unit_disp_add_AdaIn_lvl2,
    UPIR_LDR_laplacian_unit_disp_add_AdaIn_lvl3,
    SpatialTransform_unit
)
from Functions import Validation_Brats_with_mask, jacobian_determinant, generate_grid_unit

def compute_tre(warped_vox, fixed_vox):
    """Compute Euclidean distance between warped and fixed landmarks (mm)."""
    return np.linalg.norm(np.asarray(warped_vox) - np.asarray(fixed_vox), axis=1)

# ---------------------------
# Evaluate fold
# ---------------------------
def evaluate_fold(model, fixed_list, moving_list, fixed_csv_list, moving_csv_list, tumor_seg_list, uncertainty_activation, transform):
    Jaco30 = []
    Jaco = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {'tumor_tre': [], 'nontumor_tre': [], 
               'tumor_robust': [], 'nontumor_robust': [],
               'Jaco30': [], 'Jaco': [], 'inference_time': []}  # Include inference_time in results

    dataset = Validation_Brats_with_mask(fixed_list, moving_list, fixed_csv_list, moving_csv_list, tumor_seg_list, norm=True)
    dataloader = Data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load template for header/affine
    template = nib.load(fixed_list[0])
    header, affine = template.header, template.affine

    for batch_idx, data in enumerate(dataloader):
        Y_ori, X_ori = data['move'].to(device), data['fixed'].to(device)
        X_label, Y_label = data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
        tumor_mask = data['tumor_mask'].to(device)
        tumor_seg_path = tumor_seg_list[batch_idx]
        tumor_seg_nii = nib.load(tumor_seg_path)
        tumor_seg = binary_dilation(tumor_seg_nii.get_fdata(), iterations=30)

        ori_img_shape = X_ori.shape[2:]
        h, w, d = ori_img_shape

        X = F.interpolate(X_ori, size=imgshape, mode='trilinear')
        Y = F.interpolate(Y_ori, size=imgshape, mode='trilinear')

        with torch.no_grad():
            reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(0)

            # Measure inference time
            start_time = time.time()
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, _, _= model(X, Y, reg_code)
            inference_time = time.time() - start_time
            
            results['inference_time'].append(inference_time)  # Store inference time

            F_X_Y = F.interpolate(F_X_Y, size=ori_img_shape, mode='trilinear', align_corners=True)

            grid_unit = generate_grid_unit(ori_img_shape)
            grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).to(device).float()

            # Warp image
            X_Y = transform(X_ori, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)

            # Full displacement
            full_F_X_Y = torch.zeros(F_X_Y.shape)
            full_F_X_Y[0, 0] = F_X_Y[0, 2] * (h - 1) / 2
            full_F_X_Y[0, 1] = F_X_Y[0, 1] * (w - 1) / 2
            full_F_X_Y[0, 2] = F_X_Y[0, 0] * (d - 1) / 2
            
            full_F_X_Y_np = full_F_X_Y.cpu().numpy()[0]
            jdet = jacobian_determinant(full_F_X_Y).astype(np.float16)

            jdet = np.pad(jdet, ((2, 2), (2, 2), (2, 2)), mode='constant', constant_values=1).astype(np.float16)

            # Jacobian
            inv_seg = ((1 - tumor_seg) > 0) & (X_Y.cpu().numpy()[0, 0] > 0)
            nontumor_jdet = jdet[inv_seg]
            Jaco.append(100 * (nontumor_jdet < 0).sum() / inv_seg.sum())

            if tumor_seg.sum() != 0:
                tumor_reg = tumor_seg > 0
                tumor_jdet = jdet[tumor_reg]
                Jaco30.append(100 * (tumor_jdet < 0).sum() / tumor_seg.sum())
            else:
                Jaco30.append(0)

            # Split landmarks
            Y_label_tumor, X_label_tumor = [], []
            Y_label_nontumor, X_label_nontumor = [], []

            for i in range(Y_label.shape[0]):
                if tumor_seg[int(Y_label[i, 0]), int(Y_label[i, 1]), int(Y_label[i, 2])] == 1:
                    Y_label_tumor.append(Y_label[i])
                    X_label_tumor.append(X_label[i])
                else:
                    Y_label_nontumor.append(Y_label[i])
                    X_label_nontumor.append(X_label[i])

            Y_label_tumor = np.array(Y_label_tumor)
            X_label_tumor = np.array(X_label_tumor)
            Y_label_nontumor = np.array(Y_label_nontumor)
            X_label_nontumor = np.array(X_label_nontumor)

            # Tumor TRE & robustness
            if len(Y_label_tumor) > 0:
                moving_disp = np.stack([map_coordinates(full_F_X_Y_np[i], X_label_tumor.T) for i in range(3)], axis=1)
                warped_moving = X_label_tumor + moving_disp
                tre_score = compute_tre(warped_moving, Y_label_tumor)
                ori_tre = compute_tre(X_label_tumor, Y_label_tumor)
                robustness = (tre_score < ori_tre).mean()
                tumor_tre = tre_score.mean()
            else:
                tumor_tre, robustness = 0, 1

            # Non-tumor TRE & robustness
            if len(Y_label_nontumor) > 0:
                moving_disp = np.stack([map_coordinates(full_F_X_Y_np[i], X_label_nontumor.T) for i in range(3)], axis=1)
                warped_moving = X_label_nontumor + moving_disp
                tre_score = compute_tre(warped_moving, Y_label_nontumor)
                ori_tre = compute_tre(X_label_nontumor, Y_label_nontumor)
                nontumor_robust = (tre_score < ori_tre).mean()
                nontumor_tre = tre_score.mean()
            else:
                nontumor_tre, nontumor_robust = 0, 1

            # Append results
            results['tumor_tre'].append(tumor_tre)
            results['nontumor_tre'].append(nontumor_tre)
            results['tumor_robust'].append(robustness)
            results['nontumor_robust'].append(nontumor_robust)
            results['Jaco30'].append(Jaco30[-1])
            results['Jaco'].append(Jaco[-1])

            print(f"Case {batch_idx+1}: Tumor TRE {tumor_tre:.2f}, Non-tumor TRE {nontumor_tre:.2f}, Tumor Jaco {Jaco30[-1]:.3f}, Non-tumor Jaco {Jaco[-1]:.3f}, Inference Time: {inference_time:.3f}s")

    return results

# ---------------------------
# Final results printing
# ---------------------------
def safe_stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0: return "N/A","N/A","N/A","N/A"
    return f"{arr.mean():.2f} ± {arr.std():.2f}", f"{np.median(arr):.2f}", f"{arr.min():.2f}", f"{arr.max():.2f}"

def print_final_results(all_results, out_json='evaluation_results.json', total_expected=None):
    """
    Print and save summary metrics across all folds/cases.
    Compatible with the new evaluate_fold outputs.
    """
    print("\n" + "="*72)
    print("FINAL EVALUATION RESULTS (ALL FOLDS & CASES)")
    print("="*72 + "\n")

    # Convert to numpy arrays
    metrics = {
        'Tumor TRE (mm)': np.array(all_results.get('tumor_tre', []), dtype=float),
        'Non-Tumor TRE (mm)': np.array(all_results.get('nontumor_tre', []), dtype=float),
        'Tumor Robustness': np.array(all_results.get('tumor_robust', []), dtype=float),
        'Non-Tumor Robustness': np.array(all_results.get('nontumor_robust', []), dtype=float),
        'Tumor %|J|<=0': np.array(all_results.get('Jaco30', []), dtype=float),
        'Non-Tumor %|J|<=0': np.array(all_results.get('Jaco', []), dtype=float),
        'Inference Time (s)': np.array(all_results.get('inference_time', []), dtype=float)  # Include inference time
    }

    # Print per-metric counts
    print("Number of values per metric:")
    for k, arr in metrics.items():
        print(f"  {k:<25}: {len(arr)}")

    # Optional sanity check
    if total_expected is not None:
        for k, arr in metrics.items():
            if len(arr) != total_expected:
                print(f"⚠️ Warning: {k} has {len(arr)} values, expected {total_expected}")

    print("\n" + "-"*72)
    print(f"{'Metric':<25} {'Mean ± Std':<18} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-"*72)

    # Compute and print all stats
    for name, arr in metrics.items():
        mean_std, med, minv, maxv = safe_stats(arr)
        print(f"{name:<25} {mean_std:<18} {med:<8} {minv:<8} {maxv:<8}")

    # Combined TRE across all cases
    combined = np.concatenate([metrics['Tumor TRE (mm)'], metrics['Non-Tumor TRE (mm)']]) \
        if metrics['Tumor TRE (mm)'].size and metrics['Non-Tumor TRE (mm)'].size else np.array([])
    if combined.size:
        print(f"\n{'Overall TRE (mm)':<25} {combined.mean():.2f} ± {combined.std():.2f}")
    else:
        print(f"\n{'Overall TRE (mm)':<25} N/A")

    # Inference time statistics
    if 'inference_time' in all_results and all_results['inference_time']:
        total_inference_time = np.sum(all_results['inference_time'])
        avg_inference_time = np.mean(all_results['inference_time'])
        print(f"\n{'Total Inference Time (s)':<25} {total_inference_time:.2f} seconds")
        print(f"{'Average Inference Time (s)':<25} {avg_inference_time:.3f} seconds")
        print(f"{'FPS (frames per second)':<25} {1.0 / avg_inference_time:.2f}")

    # Save to JSON (raw values)
    with open(out_json, 'w') as f:
        json.dump({k: list(map(float, v)) for k, v in all_results.items()}, f, indent=2)
    print(f"\n✅ Saved detailed results to: {out_json}")
    print("="*72)

# ---------------------------
# Main evaluation driver
# ---------------------------
def evaluate():
    datapath = "/workspace/DIRAC/Data/BraTSReg_self_train"
    fixed_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
    moving_t1ce_list = sorted([p for p in sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz")) if p not in fixed_t1ce_list])
    fixed_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarks.csv"))
    moving_csv_list = sorted([p for p in sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarks.csv")) if p not in fixed_csv_list])
    tumor_seg_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_seg.nii.gz"))

    train_val_indices = np.arange(140)
    all_results = {k: [] for k in ['tumor_tre', 'nontumor_tre', 'tumor_robust', 'nontumor_robust', 'Jaco30', 'Jaco', 'inference_time']}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_val_indices)):
        print(f"\n=== Fold {fold+1}/5 ===")
        test_fixed = [fixed_t1ce_list[i] for i in train_val_indices[test_idx]]
        test_moving = [moving_t1ce_list[i] for i in train_val_indices[test_idx]]
        test_fixed_csv = [fixed_csv_list[i] for i in train_val_indices[test_idx]]
        test_moving_csv = [moving_csv_list[i] for i in train_val_indices[test_idx]]
        test_tumor_seg = [tumor_seg_list[i] for i in train_val_indices[test_idx]]
        if fold==0:
            model_path = f'/workspace/DIRAC/Model/UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github/1UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github_stagelvl3_92000.pth'
        elif fold==1:
            model_path = f'/workspace/DIRAC/Model/UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github/2UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github_stagelvl3_124000.pth'
        elif fold==2:
            model_path = f'/workspace/DIRAC/Model/UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github/3UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github_stagelvl3_46000.pth'
        elif fold==3:
            model_path = f'/workspace/DIRAC/Model/UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github/4UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github_stagelvl3_130000.pth'
        else:
            model_path = f'/workspace/DIRAC/Model/UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github/5UPIR_NCC_fea6b5_AdaIn64_t1ce_fbcon_aug_mean_fffixed_github_stagelvl3_96000.pth'
        if not os.path.exists(model_path):
            print("Model not found:", model_path)
            continue
        
        model_lvl1 = UPIR_LDR_laplacian_unit_disp_add_AdaIn_lvl1(2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow, num_block=num_cblock).cuda()
        model_lvl2 = UPIR_LDR_laplacian_unit_disp_add_AdaIn_lvl2(2, 3, start_channel, is_train=True, imgshape=imgshape_2, range_flow=range_flow, model_lvl1=model_lvl1, num_block=num_cblock).cuda()
        model = UPIR_LDR_laplacian_unit_disp_add_AdaIn_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape, range_flow=range_flow, model_lvl2=model_lvl2, num_block=num_cblock).cuda()

        model.load_state_dict(torch.load(model_path))
        model.eval()

        fold_results = evaluate_fold(model, test_fixed, test_moving, test_fixed_csv, test_moving_csv, test_tumor_seg, torch.nn.Softplus(), SpatialTransform_unit().cuda())
        print_final_results(fold_results, out_json=f'evaluation_results_fold{fold+1}.json')

        for k in all_results: all_results[k].extend(fold_results[k])
        print(f"Completed fold {fold+1}")

    print_final_results(all_results)

# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    # Training parameters you used
    img_h, img_w, img_d = 160, 160, 80
    imgshape = (img_h, img_w, img_d)
    imgshape_4 = (img_h // 4, img_w // 4, img_d // 4)
    imgshape_2 = (img_h // 2, img_w // 2, img_d // 2)
    start_channel = 8
    range_flow = 0.4
    num_cblock = 5

    evaluate()
