import glob
import os
import random
from argparse import ArgumentParser

import math
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from scipy.ndimage.interpolation import map_coordinates

from Functions import Dataset_bratsreg_bidirection, Validation_Brats, \
    generate_grid_unit
from bratsreg_model_original import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, \
    Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3, \
    SpatialTransform_unit, smoothloss, multi_resolution_NCC_weight
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def checkerboard(img1, img2, patch_size=20):
    """Creates a checkerboard image from two input images."""
    h, w = img1.shape
    checkerboard_img = np.zeros_like(img1)
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if (i // patch_size + j // patch_size) % 2 == 0:
                checkerboard_img[i:i+patch_size, j:j+patch_size] = img1[i:i+patch_size, j:j+patch_size]
            else:
                checkerboard_img[i:i+patch_size, j:j+patch_size] = img2[i:i+patch_size, j:j+patch_size]
    return checkerboard_img

def normalize_slice(img_slice):
    """Scales a 2D image slice to the [0, 1] range."""
    min_val = img_slice.min()
    max_val = img_slice.max()
    if max_val - min_val > 1e-6: # Avoid division by zero
        return (img_slice - min_val) / (max_val - min_val)
    else:
        return img_slice - min_val # Return a zero image if it's flat
                                    # x_in = torch.cat((X, Y),dim=1)
parser = ArgumentParser()
parser.add_argument("--modelname", type=str,
                    dest="modelname",
                    default='Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_',
                    help="Model name")
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=130001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: suggested range 1 to 1000000")
parser.add_argument("--occ", type=float,
                    dest="occ", default=0.01,
                    help="Mask loss: suggested range 0.01 to 1")
parser.add_argument("--inv_con", type=float,
                    dest="inv_con", default=0.1,
                    help="Inverse consistency loss: suggested range 1 to 10")
# parser.add_argument("--grad_sim", type=float,
#                     dest="grad_sim", default=0.1,
#                     help="grad_sim loss: suggested range ... to ...")
# parser.add_argument("--smooth", type=float,
#                     dest="smooth", default=12.0,
#                     help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=2000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,  # default:8, 7 for stage
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/workspace/DIRAC/Data/BraTSReg/BraTSReg_Training_Data_v2',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=3000,
                    help="Number step for freezing the previous level")
parser.add_argument("--num_cblock", type=int,
                    dest="num_cblock", default=5,
                    help="Number of conditional block")


def affine_aug(im, im_label=None):
    # mode = 'bilinear' or 'nearest'
    with torch.no_grad():
        angle_range = 5
        trans_range = 0.05
        scale_range = 0.0
        # scale_range = 0.15

        angle_xyz = (random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180,
                     random.uniform(-angle_range, angle_range) * math.pi / 180)
        scale_xyz = (random.uniform(-scale_range, scale_range), random.uniform(-scale_range, scale_range),
                     random.uniform(-scale_range, scale_range))
        trans_xyz = (random.uniform(-trans_range, trans_range), random.uniform(-trans_range, trans_range),
                     random.uniform(-trans_range, trans_range))

        rotation_x = torch.tensor([
            [1., 0, 0, 0],
            [0, math.cos(angle_xyz[0]), -math.sin(angle_xyz[0]), 0],
            [0, math.sin(angle_xyz[0]), math.cos(angle_xyz[0]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_y = torch.tensor([
            [math.cos(angle_xyz[1]), 0, math.sin(angle_xyz[1]), 0],
            [0, 1., 0, 0],
            [-math.sin(angle_xyz[1]), 0, math.cos(angle_xyz[1]), 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        rotation_z = torch.tensor([
            [math.cos(angle_xyz[2]), -math.sin(angle_xyz[2]), 0, 0],
            [math.sin(angle_xyz[2]), math.cos(angle_xyz[2]), 0, 0],
            [0, 0, 1., 0],
            [0, 0, 0, 1.]
        ], requires_grad=False).unsqueeze(0).cuda()

        trans_shear_xyz = torch.tensor([
            [1. + scale_xyz[0], 0, 0, trans_xyz[0]],
            [0, 1. + scale_xyz[1], 0, trans_xyz[1]],
            [0, 0, 1. + scale_xyz[2], trans_xyz[2]],
            [0, 0, 0, 1]
        ], requires_grad=False).unsqueeze(0).cuda()

        theta_final = torch.matmul(rotation_x, rotation_y)
        theta_final = torch.matmul(theta_final, rotation_z)
        theta_final = torch.matmul(theta_final, trans_shear_xyz)

        output_disp_e0_v = F.affine_grid(theta_final[:, 0:3, :], im.shape, align_corners=True)

        im = F.grid_sample(im, output_disp_e0_v, mode='bilinear', padding_mode="border", align_corners=True)

        if im_label is not None:
            im_label = F.grid_sample(im_label, output_disp_e0_v, mode='bilinear', padding_mode="border",
                                     align_corners=True)
            return im, im_label
        else:
            return im


def compute_tre(x, y, spacing=(1, 1, 1)):
    return np.linalg.norm((x - y) * spacing, axis=1)


def train():
    print("Training lvl3...")
    fixed_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
    moving_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz"))
    moving_t1ce_list = sorted([path for path in moving_t1ce_list if path not in fixed_t1ce_list])
    
    fixed_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_landmarks.csv"))
    moving_csv_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_landmarks.csv"))
    moving_csv_list = sorted([path for path in moving_csv_list if path not in fixed_csv_list])
    
    # Split into two parts: first 140 for K-Fold, last 20 for training only
    train_val_indices = np.arange(140)  # Indices for the first 140
    train_only_indices = np.arange(140, 160)  # Indices for the last 20

    # Create a list of indices for K-Fold on the first 140
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    start_fold = 4  # Start from second fold

    for fold, (train_index, val_index) in enumerate(kf.split(train_val_indices)):
        if fold < start_fold:
            print(f"Skipping fold {fold + 1}")
            continue
        print(f"Fold {fold + 1}/{kf.get_n_splits()}")
        
        # Prepare training and validation sets
        train_fixed = [fixed_t1ce_list[i] for i in train_val_indices[train_index]]
        train_moving = [moving_t1ce_list[i] for i in train_val_indices[train_index]]
        val_fixed = [fixed_t1ce_list[i] for i in train_val_indices[val_index]]
        val_moving = [moving_t1ce_list[i] for i in train_val_indices[val_index]]
        val_fixed_csv_list = [fixed_csv_list[i] for i in train_val_indices[val_index]]
        val_moving_csv_list = [moving_csv_list[i] for i in train_val_indices[val_index]]

        # Remaining 20 for training only
        train_fixed.extend([fixed_t1ce_list[i] for i in train_only_indices])
        train_moving.extend([moving_t1ce_list[i] for i in train_only_indices])
        
        model_lvl1 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape_4,
                                                                    range_flow=range_flow, num_block=num_cblock).cuda()
        model_lvl2 = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2(2, 3, start_channel, is_train=True,
                                                                    imgshape=imgshape_2,
                                                                    range_flow=range_flow, model_lvl1=model_lvl1,
                                                                    num_block=num_cblock).cuda()

        model = Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                                                range_flow=range_flow, model_lvl2=model_lvl2,
                                                                num_block=num_cblock).cuda()

        # loss_similarity = mse_loss
        # loss_similarity = NCC()
        # loss_similarity = Edge_enhanced_CC()
        # loss_similarity = CC()
        # loss_similarity = Normalized_Gradient_Field_mask()
        loss_similarity = multi_resolution_NCC_weight(win=7, scale=3)

        # loss_similarity_grad = Gradient_CC()
        # loss_similarity = NCC()

        # loss_inverse = mse_loss
        # loss_antifold = antifoldloss
        loss_smooth = smoothloss
        # loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().cuda()
        # transform_nearest = SpatialTransformNearest_unit().cuda()
        # diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        # com_transform = CompositionTransform().cuda()

        for param in transform.parameters():
            param.requires_grad = False
            param.volatile = True

        # OASIS
        # names = sorted(glob.glob(datapath + '/*.nii'))[0:255]

        # # LPBA
        # names = sorted(glob.glob(datapath + '/S*_norm.nii'))[0:30]

        # grid = generate_grid(imgshape)
        # grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

        grid_unit = generate_grid_unit(imgshape)
        grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        model_dir = '/workspace/DIRAC/Model/' + model_name[0:-1]

        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        lossall = np.zeros((4, iteration_lvl3 + 1))

        training_generator = Data.DataLoader(Dataset_bratsreg_bidirection(train_fixed, train_moving, norm=True),
                                            batch_size=1,
                                            shuffle=True, num_workers=2)
        step = 0
        if fold==4 and step==0:
            load_model = True
            if load_model is True:
                model_path = "/workspace/DIRAC/Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_5stagelvl3_82000.pth"
                print("Loading weight: ", model_path)
                step = 82000
                model.load_state_dict(torch.load(model_path))
                temp_lossall = np.load("/workspace/DIRAC/Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github/lossBrats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_5stagelvl3_82000.npy")
                lossall[:, 0:82000] = temp_lossall[:, 0:82000]

        while step <= iteration_lvl3:
            for X, Y in training_generator:

                X = X.cuda().float()
                Y = Y.cuda().float()

                aug_flag = random.uniform(0, 1)
                if aug_flag > 0.2:
                    X = affine_aug(X)

                aug_flag = random.uniform(0, 1)
                if aug_flag > 0.2:
                    Y = affine_aug(Y)

                X = F.interpolate(X, size=imgshape, mode='trilinear')
                Y = F.interpolate(Y, size=imgshape, mode='trilinear')

                reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

                # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
                # lap
                F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

                F_Y_X, Y_X, X_4x, F_yx, F_yx_lvl1, F_yx_lvl2, _ = model(Y, X, reg_code)

                F_X_Y_warpped = transform(F_X_Y, F_Y_X.permute(0, 2, 3, 4, 1), grid_unit)
                F_Y_X_warpped = transform(F_Y_X, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit)

                diff_fw = F_X_Y + F_Y_X_warpped  # Y
                diff_bw = F_Y_X + F_X_Y_warpped  # X

                fw_mask = (Y_4x > 0).float()
                bw_mask = (X_4x > 0).float()

                u_diff_fw = torch.sum(torch.norm(diff_fw * fw_mask, dim=1, keepdim=True)) / torch.sum(fw_mask)
                u_diff_bw = torch.sum(torch.norm(diff_bw * bw_mask, dim=1, keepdim=True)) / torch.sum(bw_mask)

                thresh_fw = (u_diff_fw + 0.015) * torch.ones_like(Y_4x, device=Y_4x.device)
                thresh_bw = (u_diff_bw + 0.015) * torch.ones_like(X_4x, device=X_4x.device)

                # smoothing
                norm_diff_fw = torch.norm(diff_fw, dim=1, keepdim=True)
                norm_diff_bw = torch.norm(diff_bw, dim=1, keepdim=True)

                smo_norm_diff_fw = F.avg_pool3d(F.avg_pool3d(norm_diff_fw, kernel_size=5, stride=1, padding=2),
                                                kernel_size=5, stride=1, padding=2)
                smo_norm_diff_bw = F.avg_pool3d(F.avg_pool3d(norm_diff_bw, kernel_size=5, stride=1, padding=2),
                                                kernel_size=5, stride=1, padding=2)

                occ_xy = (smo_norm_diff_fw > thresh_fw).float()  # y mask
                occ_yx = (smo_norm_diff_bw > thresh_bw).float()  # x mask

                occ_xy_l = F.relu(smo_norm_diff_fw - thresh_fw) * 10.
                occ_yx_l = F.relu(smo_norm_diff_bw - thresh_bw) * 10.

                # mask occ
                occ_xy = occ_xy * fw_mask
                occ_yx = occ_yx * bw_mask

                mask_xy = 1. - occ_xy
                mask_yx = 1. - occ_yx

                # 3 level deep supervision NCC
                loss_multiNCC = loss_similarity(X_Y, Y_4x, mask_xy) + loss_similarity(Y_X, X_4x, mask_yx)

                loss_inverse = torch.mean(norm_diff_fw * mask_xy) + torch.mean(norm_diff_bw * mask_yx)
                loss_occ = torch.mean(occ_xy_l) + torch.mean(occ_yx_l)

                # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
                # loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

                # reg2 - use velocity
                _, _, x, y, z = F_X_Y.shape
                norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
                norm_vector[0, 0, 0, 0, 0] = z
                norm_vector[0, 1, 0, 0, 0] = y
                norm_vector[0, 2, 0, 0, 0] = x
                loss_regulation = loss_smooth(F_X_Y * norm_vector) + loss_smooth(F_Y_X * norm_vector)

                loss = (1. - reg_code) * loss_multiNCC + reg_code * loss_regulation + occ * loss_occ + inv_con * loss_inverse

                optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizer.step()  # apply gradients

                lossall[:, step] = np.array(
                    [loss.item(), loss_multiNCC.item(), loss_inverse.item(), loss_regulation.item()])
                sys.stdout.write(
                    "\r" + 'step "{0}" -> training loss "{1:.4f}" -sim_NCC "{2:4f}" -inv "{3:.4f}" -occ "{4:4f}" -smo "{5:.4f} -reg_c "{6:.4f}"'.format(
                        step, loss.item(), loss_multiNCC.item(), loss_inverse.item(), loss_occ.item(),
                        loss_regulation.item(), reg_code[0].item()))
                sys.stdout.flush()

                # with lr 1e-3 + with bias
                if (step % n_checkpoint == 0):
                    modelname = model_dir + '/' + model_name + f"{fold + 1}stagelvl3_" + str(step) + '.pth'
                    torch.save(model.state_dict(), modelname)
                    np.save(model_dir + '/loss' + model_name + f"{fold + 1}stagelvl3_" + str(step) + '.npy', lossall)

                    # Validation
                    # val_datapath = '/workspace/DIRAC/Data/BraTSReg/BraTSReg_validation'
                    # start, end = 0, 20
                    # val_fixed_csv_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_landmarks.csv"))
                    # val_moving_csv_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_landmarks.csv"))
                    # val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])

                    # val_fixed_t1ce_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
                    # val_moving_t1ce_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_t1ce.nii.gz"))
                    # val_moving_t1ce_list = sorted(
                    #     [path for path in val_moving_t1ce_list if path not in val_fixed_t1ce_list])

                    # assert len(val_fixed_list) == len(val_moving_list)

                    valid_generator = Data.DataLoader(
                        Validation_Brats(val_fixed, val_moving, val_fixed_csv_list,
                                        val_moving_csv_list, norm=True), batch_size=1,
                        shuffle=False, num_workers=2)

                    use_cuda = True
                    device = torch.device("cuda" if use_cuda else "cpu")
                    # dice_total = []
                    tre_total = []
                    print("\nValiding...")
                    robustness_scores = []
                    for batch_idx, data in enumerate(valid_generator):
                        # X, Y, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                        #                          data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
                        Y, X, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                                                data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
                        ori_img_shape = X.shape[2:]
                        h, w, d = ori_img_shape

                        X = F.interpolate(X, size=imgshape, mode='trilinear')
                        Y = F.interpolate(Y, size=imgshape, mode='trilinear')

                        with torch.no_grad():
                            reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
                            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y, reg_code)

                            # X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit).data.cpu().numpy()[0, 0, :, :, :]
                            # Y_label = Y_label.data.cpu().numpy()[0, 0, :, :, :]

                            F_X_Y = F.interpolate(F_X_Y, size=ori_img_shape, mode='trilinear', align_corners=True)

                            full_F_X_Y = torch.zeros(F_X_Y.shape)
                            full_F_X_Y[0, 0] = F_X_Y[0, 2] * (h - 1) / 2
                            full_F_X_Y[0, 1] = F_X_Y[0, 1] * (w - 1) / 2
                            full_F_X_Y[0, 2] = F_X_Y[0, 0] * (d - 1) / 2

                            # TRE
                            full_F_X_Y = full_F_X_Y.cpu().numpy()[0]

                            fixed_keypoints = Y_label
                            moving_keypoints = X_label

                            moving_disp_x = map_coordinates(full_F_X_Y[0], moving_keypoints.transpose())
                            moving_disp_y = map_coordinates(full_F_X_Y[1], moving_keypoints.transpose())
                            moving_disp_z = map_coordinates(full_F_X_Y[2], moving_keypoints.transpose())
                            lms_moving_disp = np.array((moving_disp_x, moving_disp_y, moving_disp_z)).transpose()

                            warped_moving_keypoint = moving_keypoints + lms_moving_disp

                            tre_score = compute_tre(warped_moving_keypoint, fixed_keypoints,
                                                    spacing=(1., 1., 1.)).mean()
                            tre_total.append(tre_score)
                            print(f"Generating visualizations for step {step}...")
                            # Interpolate warped image to original size for visualization
                            full_X_Y = F.interpolate(X_Y, size=ori_img_shape, mode='trilinear', align_corners=False)
                            full_moving = F.interpolate(X, size=ori_img_shape, mode='trilinear', align_corners=False)
                            full_fixed = F.interpolate(Y, size=ori_img_shape, mode='trilinear', align_corners=False)

                            # Add this right after your TRE computation and before the other visualizations

                            # --- Landmark Visualization ---
                            print(f"Generating landmark visualizations for step {step}...")

                            # Select a central slice for visualization
                            z_slice = d // 4  # Middle slice

                            # Get slices for visualization
                            fixed_slice = normalize_slice(Y.cpu().numpy()[0, 0, :, :, z_slice])
                            full_fixed_slice = full_fixed.cpu().numpy()[0, 0, :, :, d //2]
                            moving_slice = normalize_slice(X.cpu().numpy()[0, 0, :, :, z_slice])
                            full_moving_slice = full_moving.cpu().numpy()[0, 0, :, :, d //2]
                            warped_slice = normalize_slice(X_Y.cpu().numpy()[0, 0, :, :, z_slice])
                            full_warped_slice = full_X_Y.cpu().numpy()[0, 0, :, :, d//2]

                            # Filter landmarks that are close to this slice (within ±2 slices)
                            slice_thickness = 80
                            relevant_indices = np.where(
                                (fixed_keypoints[:, 2] >= z_slice - slice_thickness) & 
                                (fixed_keypoints[:, 2] <= z_slice + slice_thickness)
                            )[0]

                            if len(relevant_indices) > 0:
                                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                                fig.suptitle(f'Step: {step} - Landmark Alignment (Slice {z_slice}) - TRE: {tre_score:.2f}mm', fontsize=14)
                                
                                # Fixed image with fixed landmarks
                                axes[0].imshow(full_fixed_slice, cmap='gray')
                                fixed_slice_landmarks = moving_keypoints[relevant_indices]
                                axes[0].scatter(fixed_slice_landmarks[:, 1], fixed_slice_landmarks[:, 0], 
                                            c='green', s=50, marker='o', label='Fixed', alpha=0.8)
                                axes[0].set_title('Fixed Image + Landmarks')
                                axes[0].legend()
                                
                                # Moving image with moving landmarks
                                axes[1].imshow(full_moving_slice, cmap='gray')
                                moving_slice_landmarks = fixed_keypoints[relevant_indices]
                                axes[1].scatter(moving_slice_landmarks[:, 1], moving_slice_landmarks[:, 0], 
                                            c='red', s=50, marker='x', label='Moving', alpha=0.8)
                                axes[1].set_title('Moving Image + Landmarks')
                                axes[1].legend()
                                
                                # Warped image with warped landmarks and fixed landmarks
                                axes[2].imshow(full_warped_slice, cmap='gray')
                                warped_slice_landmarks = warped_moving_keypoint[relevant_indices]
                                
                                # Plot warped landmarks
                                axes[2].scatter(warped_slice_landmarks[:, 1], warped_slice_landmarks[:, 0], 
                                            c='blue', s=50, marker='s', label='Warped', alpha=0.8)
                                # Plot fixed landmarks for comparison
                                axes[2].scatter(moving_slice_landmarks[:, 1], moving_slice_landmarks[:, 0], 
                                            c='green', s=30, marker='o', label='Fixed', alpha=0.6)
                                
                                # Draw lines between corresponding points to show displacement
                                for i in range(len(relevant_indices)):
                                    idx = relevant_indices[i]
                                    axes[2].plot([warped_moving_keypoint[idx, 1], fixed_keypoints[idx, 1]],
                                                [warped_moving_keypoint[idx, 0], fixed_keypoints[idx, 0]],
                                                'y-', linewidth=1, alpha=0.6)
                                
                                axes[2].set_title('Warped + Fixed Landmarks (Yellow lines show error)')
                                axes[2].legend()
                                
                                plt.tight_layout()
                                plt.savefig(f"{model_dir}/{fold + 1}landmarks_step_{step}_batch_{batch_idx}_slice_{z_slice}.png", 
                                            dpi=150, bbox_inches='tight')
                                plt.close(fig)
                                
                                # --- Additional: Create a detailed error visualization ---
                                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                                ax.imshow(full_warped_slice, cmap='gray', alpha=0.7)
                                
                                # Plot landmarks with error vectors
                                for i in range(len(relevant_indices)):
                                    idx = relevant_indices[i]
                                    ax.arrow(warped_moving_keypoint[idx, 1], warped_moving_keypoint[idx, 0],
                                            fixed_keypoints[idx, 1] - warped_moving_keypoint[idx, 1],
                                            fixed_keypoints[idx, 0] - warped_moving_keypoint[idx, 0],
                                            head_width=3, head_length=3, fc='red', ec='red', alpha=0.8)
                                    
                                    # Add error distance text
                                    error_distance = np.sqrt((fixed_keypoints[idx, 1] - warped_moving_keypoint[idx, 1])**2 +
                                                            (fixed_keypoints[idx, 0] - warped_moving_keypoint[idx, 0])**2)
                                    ax.text(warped_moving_keypoint[idx, 1] + 5, warped_moving_keypoint[idx, 0] - 5,
                                        f'{error_distance:.1f}', color='white', fontsize=8,
                                        bbox=dict(boxstyle="round,pad=0.3", fc='red', alpha=0.7))
                                
                                ax.set_title(f'Landmark Registration Errors\nAverage TRE: {tre_score:.2f}mm')
                                plt.savefig(f"{model_dir}/{fold + 1}landmark_errors_step_{step}_batch_{batch_idx}.png",
                                            dpi=150, bbox_inches='tight')
                                plt.close(fig)

                            else:
                                print(f"No landmarks near slice {z_slice} for batch {batch_idx}")

                            # --- Continue with your other visualizations ---
                            print(f"Generating other visualizations for step {step}...")
                            # [Your existing visualization code continues here...]
                            
                            # 1. Comparison Visualization (Moving, Fixed, Warped, Difference)
                            diff_slice = np.abs(fixed_slice - warped_slice)
                            # --- 2. High-pass filter the normalized difference map ---
                            # Apply a Gaussian blur to get the low-frequency component (the ghosting)
                            # The 'sigma' value controls the amount of blurring.
                            # A larger sigma removes larger-scale variations. Start with 1 or 2.
                            from scipy.ndimage import gaussian_filter
                            low_freq_diff = gaussian_filter(diff_slice, sigma=2)
                            
                            # Subtract the low-frequency component to get the high-pass filtered result
                            high_pass_diff = diff_slice - low_freq_diff

                            # --- Plotting all results ---
                            fig, axes = plt.subplots(1, 6, figsize=(25, 5))
                            fig.suptitle(f'Step: {batch_idx} - TRE: {tre_score:.2f}mm', fontsize=16)
                            
                            # Original images
                            axes[0].imshow(moving_slice, cmap='gray'); axes[0].set_title('Moving'); axes[0].axis('off')
                            axes[1].imshow(warped_slice, cmap='gray'); axes[1].set_title('Warped'); axes[1].axis('off')
                            axes[2].imshow(fixed_slice, cmap='gray'); axes[2].set_title('Fixed'); axes[2].axis('off')
                            # Normalized Difference (still has ghosting)
                            im1 = axes[3].imshow(diff_slice, cmap='hot', vmin=0, vmax=1); axes[3].set_title('Normalized Difference'); axes[3].axis('off')
                            fig.colorbar(im1, ax=axes[3], shrink=0.8)

                            # The Low-Frequency Component We Are Removing
                            im2 = axes[4].imshow(low_freq_diff, cmap='hot', vmin=0, vmax=1); axes[4].set_title('Ghosting Artifact (Blurred)'); axes[4].axis('off')
                            fig.colorbar(im2, ax=axes[4], shrink=0.8)
                            
                            # The Final, Cleaned Error Map
                            im3 = axes[5].imshow(high_pass_diff, cmap='hot'); axes[5].set_title('High-Pass Filtered Error'); axes[5].axis('off')
                            fig.colorbar(im3, ax=axes[5], shrink=0.8)
                            
                            plt.savefig(os.path.join(f"step_{batch_idx}_filtered_comparison.png"), bbox_inches='tight')
                            plt.close(fig)

                            # 2. Checkerboard Visualization
                            checkerboard_img = checkerboard(fixed_slice, warped_slice, patch_size=w//8)
                            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                            ax.imshow(checkerboard_img, cmap='gray'); ax.set_title('Checkerboard Overlay'); ax.axis('off')
                            plt.savefig(os.path.join("step_{batch_idx}_checkerboard.png"), bbox_inches='tight')
                            plt.close(fig)

                            # 3. Warped Grid Visualization
                            disp_slice = full_F_X_Y[0:2, :, :, z_slice] # Get x and y displacements
                            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
                            warped_grid_x = grid_x + disp_slice[0]
                            warped_grid_y = grid_y + disp_slice[1]
                            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                            ax.imshow(full_warped_slice, cmap='gray')
                            for i in range(0, w, 15):
                                ax.plot(warped_grid_x[:, i], warped_grid_y[:, i], 'c-', linewidth=0.5)
                            for i in range(0, h, 15):
                                ax.plot(warped_grid_x[i, :], warped_grid_y[i, :], 'c-', linewidth=0.5)
                            ax.set_title('Warped Grid'); ax.axis('off')
                            plt.savefig(os.path.join(f"step_{batch_idx}_warped_grid.png"), bbox_inches='tight')
                            plt.close(fig)

                            # 4. Quiver Plot Visualization
                            downsample = 12 # Downsample for clarity
                            quiver_y, quiver_x = grid_y[::downsample, ::downsample], grid_x[::downsample, ::downsample]
                            disp_y, disp_x = disp_slice[1, ::downsample, ::downsample], disp_slice[0, ::downsample, ::downsample]

                            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                            ax.imshow(full_moving_slice, cmap='gray', alpha=0.7)
                            ax.quiver(quiver_x, quiver_y, disp_x, disp_y, color='r', angles='xy', scale_units='xy', scale=1, headwidth=4, width=0.005)
                            ax.set_title('Deformation Field (Quiver)'); ax.axis('off')
                            plt.savefig(os.path.join(f"step_{batch_idx}_quiver.png"), bbox_inches='tight')
                            plt.close(fig)
                            # --- Before registration ---
                            tre_before = np.linalg.norm(moving_keypoints - fixed_keypoints, axis=1)

                            # --- After registration ---
                            tre_after = np.linalg.norm(warped_moving_keypoint - fixed_keypoints, axis=1)

                            # --- Landmark-level improvement check ---
                            improved_landmarks = np.sum(tre_after < tre_before)
                            total_landmarks = len(tre_before)

                            # --- Robustness for this pair (BraTS-Reg definition) ---
                            robustness_pair = improved_landmarks / total_landmarks

                            # Store it
                            robustness_scores.append(robustness_pair)

                        # --- End of Visualization Code ---

                    # Calculate mean and standard deviation
                    tre_total = np.array(tre_total)
                    print(tre_total)
                    tre_mean = tre_total.mean()
                    tre_std = tre_total.std()
                    R_mean = np.mean(robustness_scores)
                    R_std = np.std(robustness_scores)
                    print(f"Robustness - Mean: {R_mean:.4f}, Std: {R_std:.4f}")

                    print(f"TRE - Mean: {tre_mean:.4f}mm, Std: {tre_std:.4f}mm")
                    print(f"TRE - Min: {tre_total.min():.4f}mm, Max: {tre_total.max():.4f}mm")
                    

                    # Log both mean and standard deviation
                    log_dir = f"/workspace/DIRAC/Log/{fold + 1}without_Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_.txt"
                    with open(log_dir, "a") as log:
                        log.write(f"{step}: Mean={tre_mean:.4f}, Std={tre_std:.4f}, Min={tre_total.min():.4f}, Max={tre_total.max():.4f}\n, R={R_mean:.4f}, R_std={R_std:.4f}\n")

                # if step == freeze_step:
                #     model.unfreeze_modellvl2()
                #     # num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
                #     # print("\nmodel_lvl2_num_param_requires_grad: ", num_param)

                step += 1

                if step > iteration_lvl3:
                    break
            print("one epoch pass")
        np.save(model_dir + '/loss' + model_name + f'{fold + 1}stagelvl3.npy', lossall)


if __name__ == '__main__':
    opt = parser.parse_args()

    lr = opt.lr
    start_channel = opt.start_channel
    antifold = opt.antifold
    # grad_sim = opt.grad_sim
    n_checkpoint = opt.checkpoint
    # smooth = opt.smooth
    datapath = opt.datapath
    freeze_step = opt.freeze_step
    num_cblock = opt.num_cblock
    occ = opt.occ
    inv_con = opt.inv_con

    iteration_lvl3 = opt.iteration_lvl3

    model_name = opt.modelname

    # Create and initalize log file
    if not os.path.isdir("../Log"):
        os.mkdir("../Log")

    log_dir = "../Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation TRE log for " + model_name[0:-1] + ":\n")

    img_h, img_w, img_d = 160, 160, 80
    imgshape = (img_h, img_w, img_d)
    imgshape_4 = (img_h // 4, img_w // 4, img_d // 4)
    imgshape_2 = (img_h // 2, img_w // 2, img_d // 2)

    range_flow = 0.4
    print("Training %s ..." % model_name)
    train()