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
from scipy.ndimage import median_filter
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from Functions import Dataset_bratsreg_bidirection, Validation_Brats, \
    generate_grid_unit
from bratsreg_model_stage import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, \
    Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3, Miccai2021_LDR_laplacian_TransMorph_lvl3, uncern_Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3,\
    SpatialTransform_unit, smoothloss, multi_resolution_NCC_weight, multi_resolution_NCC_weight_2D, MultiResolution_Cosine, DINO_Cosine_Similarity, DINO_Cosine_Loss, SpecialistHead
from cnn_swin import Dual_FusionMorph

from transformers import AutoImageProcessor, AutoModel

# We'll use a functional median filter from Scipy
def median_filter_3d(x, kernel_size=3):
    # The input tensor is moved to CPU and converted to a numpy array for scipy
    x_np = x.squeeze().cpu().detach().numpy()
    # The median filter is applied
    filtered_np = median_filter(x_np, size=kernel_size)
    # The numpy array is converted back to a torch tensor and moved to the correct device
    return torch.from_numpy(filtered_np).unsqueeze(0).cuda().float()

def save_visualizations(source_image, fixed_image, warped_image, uncertainty_map, filename):
    """
    Saves a 2D slice of the 3D images and the uncertainty map for visualization.
    
    Args:
        source_image (torch.Tensor): The original moving image.
        fixed_image (torch.Tensor): The original fixed image.
        warped_image (torch.Tensor): The warped moving image.
        uncertainty_map (torch.Tensor): The predicted uncertainty map.
        filename (str): The name of the file to save the visualization to.
    """
    # Detach tensors from the computation graph and move to CPU
    source_slice = source_image.squeeze().cpu().detach().numpy()[:, :, source_image.shape[-1] // 2]
    fixed_slice = fixed_image.squeeze().cpu().detach().numpy()[:, :, fixed_image.shape[-1] // 2]
    warped_slice = warped_image.squeeze().cpu().detach().numpy()[:, :, warped_image.shape[-1] // 2]
    # Get the 3 uncertainty channels and slice them
    uncertainty_x_slice = uncertainty_map.squeeze().cpu().detach().numpy()[:, :, uncertainty_map.shape[-1] // 2]
    # uncertainty_y_slice = uncertainty_map.squeeze().cpu().detach().numpy()[1, :, :, uncertainty_map.shape[-1] // 2]
    # uncertainty_z_slice = uncertainty_map.squeeze().cpu().detach().numpy()[2, :, :, uncertainty_map.shape[-1] // 2]

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot the images
    axes[0, 0].imshow(source_slice, cmap='gray')
    axes[0, 0].set_title('Source Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fixed_slice, cmap='gray')
    axes[0, 1].set_title('Fixed Image')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(warped_slice, cmap='gray')
    axes[0, 2].set_title('Warped Image')
    axes[0, 2].axis('off')

    # Plot the uncertainty maps for each dimension
    axes[1, 0].imshow(uncertainty_x_slice, cmap='hot')
    axes[1, 0].set_title('Uncertainty (X-axis)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(uncertainty_x_slice, cmap='hot')
    axes[1, 1].set_title('Uncertainty (Y-axis)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(uncertainty_x_slice, cmap='hot')
    axes[1, 2].set_title('Uncertainty (Z-axis)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    
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


class Config:
    def __init__(self):
        # Model parameters
        self.patch_size = 4
        self.in_chans = 2                  # Number of input channels
        self.embed_dim = 16                # Embedding dimension
        self.depths = (2, 2, 2, 2, 2, 2)         # Depth of each Swin Transformer layer
        self.num_heads = (4, 4, 4, 4, 4, 4)      # Number of attention heads in each layer
        self.window_size = (7, 7, 7, 7)    # Window size for self-attention
        self.mlp_ratio = 4                  # Ratio of MLP hidden dim to embedding dim
        self.pat_merg_rf = 4                # Patch merging reference factor
        self.qkv_bias = False                # If True, add a learnable bias to query, key, value
        self.drop_rate = 0                   # Dropout rate
        self.drop_path_rate = 0.3            # Stochastic depth rate
        self.ape = False                     # Absolute position embedding
        self.spe = False                     # Sinusoidal positional embedding
        self.rpe = True                      # Relative position embedding
        self.patch_norm = True               # Use normalization after patch embedding
        self.use_checkpoint = False           # Use checkpointing
        self.out_indices = (0, 1, 2, 3)     # Indices of layers to output
        self.reg_head_chan = 16             # Number of channels in the registration head
        self.img_size = (160, 160, 80)      # Input image size
        self.zernike_embed_dim = 121
        self.num_layers = 6

# Creating a config object
# config = Config()

def compute_spatial_gradients_3d(field):
    """Compute spatial gradients using central finite differences"""
    D, H, W = field.shape[2:]
    grad_x = torch.zeros_like(field)
    grad_y = torch.zeros_like(field)
    grad_z = torch.zeros_like(field)
    
    grad_x[:, :, 1:-1, :, :] = (field[:, :, 2:, :, :] - field[:, :, :-2, :, :]) / 2.0
    grad_y[:, :, :, 1:-1, :] = (field[:, :, :, 2:, :] - field[:, :, :, :-2, :]) / 2.0
    grad_z[:, :, :, :, 1:-1] = (field[:, :, :, :, 2:] - field[:, :, :, :, :-2]) / 2.0
    
    grad_x[:, :, 0, :, :] = field[:, :, 1, :, :] - field[:, :, 0, :, :]
    grad_x[:, :, -1, :, :] = field[:, :, -1, :, :] - field[:, :, -2, :, :]
    
    grad_y[:, :, :, 0, :] = field[:, :, :, 1, :] - field[:, :, :, 0, :]
    grad_y[:, :, :, -1, :] = field[:, :, :, -1, :] - field[:, :, :, -2, :]
    
    grad_z[:, :, :, :, 0] = field[:, :, :, :, 1] - field[:, :, :, :, 0]
    grad_z[:, :, :, :, -1] = field[:, :, :, :, -1] - field[:, :, :, :, -2]
    
    return grad_x, grad_y, grad_z

def compute_stress_field(displacement_field, lam=1.0, mu=1.0):
    """Compute stress field from displacement field using finite differences"""
    grad_x, grad_y, grad_z = compute_spatial_gradients_3d(displacement_field)
    
    dUx_dx, dUy_dx, dUz_dx = grad_x[:, 0:1], grad_x[:, 1:2], grad_x[:, 2:3]
    dUx_dy, dUy_dy, dUz_dy = grad_y[:, 0:1], grad_y[:, 1:2], grad_y[:, 2:3]
    dUx_dz, dUy_dz, dUz_dz = grad_z[:, 0:1], grad_z[:, 1:2], grad_z[:, 2:3]
    
    Exx = dUx_dx
    Eyy = dUy_dy
    Ezz = dUz_dz
    Exy = 0.5 * (dUx_dy + dUy_dx)
    Exz = 0.5 * (dUx_dz + dUz_dx)
    Eyz = 0.5 * (dUy_dz + dUz_dy)
    
    trace_E = Exx + Eyy + Ezz
    Sxx = lam * trace_E + 2 * mu * Exx
    Syy = lam * trace_E + 2 * mu * Eyy
    Szz = lam * trace_E + 2 * mu * Ezz
    Sxy = 2 * mu * Exy
    Sxz = 2 * mu * Exz
    Syz = 2 * mu * Eyz
    
    return torch.cat([Sxx, Syy, Szz, Sxy, Sxz, Syz], dim=1)

def compute_stress_field_nonlinear(displacement_field, lam=1.0, mu=1.0):
    """
    Compute stress field using the NON-LINEAR Green-Lagrangian strain tensor.
    This is more physically accurate for large deformations like brain shift.
    """
    # 1. Compute the displacement gradient tensor J (also known as ∇d)
    grad_x, grad_y, grad_z = compute_spatial_gradients_3d(displacement_field)
    
    # Let's stack the gradients to form the Jacobian J
    # grad_x is [dUx/dx, dUy/dx, dUz/dx]
    # grad_y is [dUx/dy, dUy/dy, dUz/dz]
    # grad_z is [dUx/dz, dUy/dz, dUz/dz]
    
    # We need to reshape for matrix multiplication later
    J_row1 = torch.cat([grad_x[:, 0:1], grad_y[:, 0:1], grad_z[:, 0:1]], dim=1)
    J_row2 = torch.cat([grad_x[:, 1:2], grad_y[:, 1:2], grad_z[:, 1:2]], dim=1)
    J_row3 = torch.cat([grad_x[:, 2:3], grad_y[:, 2:3], grad_z[:, 2:3]], dim=1)
    
    J = torch.stack([J_row1, J_row2, J_row3], dim=1) # Shape: [B, 3, 3, D, H, W]
    J_T = J.transpose(1, 2)

    # 2. Calculate the Green-Lagrangian strain tensor E
    # E = 0.5 * (J + J^T + J^T @ J)
    # E is now a tensor of symmetric 3x3 strain matrices at each voxel
    # Perform a batched matrix multiplication
    J_T_times_J = torch.matmul(J_T.permute(0, 3, 4, 5, 1, 2), J.permute(0, 3, 4, 5, 1, 2)).permute(0, 4, 5, 1, 2, 3)
    
    E = 0.5 * (J + J_T + J_T_times_J)

    # 3. Extract the 6 unique components for Hooke's Law
    Exx = E[:, 0, 0:1]
    Eyy = E[:, 1, 1:2]
    Ezz = E[:, 2, 2:3]
    Exy = E[:, 0, 1:2]
    Exz = E[:, 0, 2:3]
    Eyz = E[:, 1, 2:3]
    
    # 4. Calculate Stress using Hooke's Law
    trace_E = Exx + Eyy + Ezz
    Sxx = lam * trace_E + 2 * mu * Exx
    Syy = lam * trace_E + 2 * mu * Eyy
    Szz = lam * trace_E + 2 * mu * Ezz
    Sxy = 2 * mu * Exy
    Sxz = 2 * mu * Exz
    Syz = 2 * mu * Eyz
    
    return torch.cat([Sxx, Syy, Szz, Sxy, Sxz, Syz], dim=1)

def compute_divergence_3d(stress_tensor_field):
    """Compute divergence of a stress tensor field using finite differences"""
    Sxx_grad_x, _, _ = compute_spatial_gradients_3d(stress_tensor_field[:, 0:1])
    _, Syy_grad_y, _ = compute_spatial_gradients_3d(stress_tensor_field[:, 1:2])
    _, _, Szz_grad_z = compute_spatial_gradients_3d(stress_tensor_field[:, 2:3])
    Sxy_grad_x, Sxy_grad_y, _ = compute_spatial_gradients_3d(stress_tensor_field[:, 3:4])
    Sxz_grad_x, _, Sxz_grad_z = compute_spatial_gradients_3d(stress_tensor_field[:, 4:5])
    _, Syz_grad_y, Syz_grad_z = compute_spatial_gradients_3d(stress_tensor_field[:, 5:6])
    
    div_x = Sxx_grad_x + Sxy_grad_y + Sxz_grad_z
    div_y = Sxy_grad_x + Syy_grad_y + Syz_grad_z
    div_z = Sxz_grad_x + Syz_grad_y + Szz_grad_z
    
    return torch.cat([div_x, div_y, div_z], dim=1)

def calculate_physics_loss(deformation_field, roi_mask, num_samples=2048):
    if torch.sum(roi_mask) < 1:
        return torch.tensor(0.0, device=deformation_field.device)
    
    stress_field = compute_stress_field_nonlinear(deformation_field)
    stress_divergence = compute_divergence_3d(stress_field)
    
    roi_indices = torch.nonzero(roi_mask.squeeze(), as_tuple=False)
    if roi_indices.size(0) > num_samples:
        sampled_indices = roi_indices[torch.randperm(roi_indices.size(0))[:num_samples]]
    else:
        sampled_indices = roi_indices

    D, H, W = roi_mask.shape[2:]
    points_z = sampled_indices[:, 0].float()
    points_y = sampled_indices[:, 1].float()
    points_x = sampled_indices[:, 2].float()

    grid_z_norm = (points_z / (D - 1)) * 2 - 1
    grid_y_norm = (points_y / (H - 1)) * 2 - 1
    grid_x_norm = (points_x / (W - 1)) * 2 - 1
    
    sample_grid = torch.stack([grid_x_norm, grid_y_norm, grid_z_norm], dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    divergence_samples = F.grid_sample(stress_divergence, sample_grid, mode='bilinear', align_corners=False).squeeze()
    
    if divergence_samples.dim() == 1:
        divergence_samples = divergence_samples.unsqueeze(0)
        
    loss = torch.mean(torch.norm(divergence_samples, dim=1) ** 2)
    return loss


def calculate_equilibrium_loss(predicted_stress_field, roi_mask, num_samples=1024):
    """
    Calculates the static equilibrium loss.
    It takes the STRESS field PREDICTED by the model and checks if it's physically stable (divergence-free).
    """
    if torch.sum(roi_mask) < 1:
        return torch.tensor(0.0, device=predicted_stress_field.device)
    
    # The stress field is now an INPUT, not calculated internally.
    # The compute_stress_field() call is REMOVED.
    
    # Compute the divergence of the PREDICTED stress field
    stress_divergence = compute_divergence_3d(predicted_stress_field)
    
    # The rest of the function (sampling and loss calculation) remains the same.
    roi_indices = torch.nonzero(roi_mask.squeeze(), as_tuple=False)
    if roi_indices.size(0) > num_samples:
        sampled_indices = roi_indices[torch.randperm(roi_indices.size(0))[:num_samples]]
    else:
        sampled_indices = roi_indices

    D, H, W = roi_mask.shape[2:]
    points_z = sampled_indices[:, 0].float()
    points_y = sampled_indices[:, 1].float()
    points_x = sampled_indices[:, 2].float()

    grid_z_norm = (points_z / (D - 1)) * 2 - 1
    grid_y_norm = (points_y / (H - 1)) * 2 - 1
    grid_x_norm = (points_x / (W - 1)) * 2 - 1
    
    sample_grid = torch.stack([grid_x_norm, grid_y_norm, grid_z_norm], dim=1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    divergence_samples = F.grid_sample(stress_divergence, sample_grid, mode='bilinear', align_corners=False).squeeze()
    
    if divergence_samples.dim() == 1:
        divergence_samples = divergence_samples.unsqueeze(0)
        
    # The loss is the L2 norm of the stress divergence (the net force)
    loss = torch.mean(torch.norm(divergence_samples, dim=1) ** 2)
    return loss

def calculate_local_ncc(I, J, win=5, eps=1e-8, channel=1):
    """
    Calculates the local normalized cross-correlation (NCC) map.
    This is used to create a supervision target that is robust to intensity differences.
    """
    ndims = 3
    win_size = win
    
    weight = torch.ones((channel, 1, win_size, win_size, win_size), device=I.device, requires_grad=False)
    conv_fn = F.conv3d

    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, weight, padding=int(win_size / 2), groups=channel)
    J_sum = conv_fn(J, weight, padding=int(win_size / 2), groups=channel)
    I2_sum = conv_fn(I2, weight, padding=int(win_size / 2), groups=channel)
    J2_sum = conv_fn(J2, weight, padding=int(win_size / 2), groups=channel)
    IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2), groups=channel)
    
    win_prod = np.prod([win] * ndims)
    u_I = I_sum / win_prod
    u_J = J_sum / win_prod

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_prod
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_prod
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_prod
    
    # Calculate the local correlation map
    cc = cross * cross / (I_var * J_var + eps)
    
    # Return the local correlation map
    return cc

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

    # Iterate over each fold
    for fold, (train_index, val_index) in enumerate(kf.split(train_val_indices)):
        # fold=1
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

        model = uncern_Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3(2, 3, start_channel, is_train=True, imgshape=imgshape,
                                                                range_flow=range_flow, model_lvl2=model_lvl2,
                                                                num_block=num_cblock).cuda()
        # model = Miccai2021_LDR_laplacian_TransMorph_lvl3(is_train=True, imgshape=imgshape,
        #                                                 range_flow=range_flow, model_lvl2=model_lvl2).cuda()
        # --- START: NEW PERCEPTUAL LOSS SETUP ---
        print("Loading pre-trained model for perceptual loss...")
        # NOTE: "vit7b16" is a massive model. Consider starting with a smaller one
        # like "facebook/dinov2-vits14" or "facebook/dinov3-convnext-tiny..."
        # if you encounter memory issues.
        pretrained_model_name = "facebook/dinov2-base"
        dino_processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        dino_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
        dino_model.eval() # Set to evaluation mode
        loss_correspondence = DINO_Cosine_Loss()
        # specialist_head = SpecialistHead().cuda()
        # CRUCIAL: Freeze the DINO model weights
        for param in dino_model.parameters():
            param.requires_grad = False
            
        num_slices_for_loss = 32
        # model = Dual_FusionMorph(config).cuda()

        # loss_similarity = mse_loss
        # loss_similarity = NCC()
        # loss_similarity = Edge_enhanced_CC()
        # loss_similarity = CC()
        # loss_similarity = Normalized_Gradient_Field_mask()
        loss_similarity = multi_resolution_NCC_weight(win=7, scale=3)
        
        loss_uncertainty = torch.nn.GaussianNLLLoss()

        # loss_similarity_grad = Gradient_CC()
        # loss_similarity = NCC()

        # loss_inverse = mse_loss
        # loss_antifold = antifoldloss
        loss_smooth = smoothloss
        # loss_perceptual = multi_resolution_NCC_weight_2D(win=5, channel=768)
        # loss_Jdet = neg_Jdet_loss

        transform = SpatialTransform_unit().cuda()
        # transform_nearest = SpatialTransformNearest_unit().cuda()
        # diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        # com_transform = CompositionTransform().cuda()

        for param in transform.parameters():
            param.requires_grad = False

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

        training_generator = Data.DataLoader(Dataset_bratsreg_bidirection(fixed_t1ce_list, moving_t1ce_list, norm=True),
                                            batch_size=1,
                                            shuffle=True, num_workers=2)
        step = 0
        load_model = True
        if load_model is True:
            model_path = "/workspace/DIRAC/Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github/1Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_stagelvl3_68000.pth"
            print("Loading weight: ", model_path)
            step = 68000
            model.load_state_dict(torch.load(model_path))
            # specialist_head.load_state_dict(torch.load(os.path.join(model_dir, f'specialist_head_step_{step}.pth')))
            temp_lossall = np.load("/workspace/DIRAC/Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github/loss1Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_stagelvl3_68000.npy")
            lossall[:, 0:68000] = temp_lossall[:, 0:68000]

        # Add sigmoid activation to ensure uncertainty is in [0,1]
        uncertainty_activation = torch.nn.Softplus()

        while step < iteration_lvl3:
            for X, Y in training_generator:
                
                X = X.cuda().float()
                Y = Y.cuda().float()
                    
                # (Original data augmentation and interpolation code)
                aug_flag = random.uniform(0, 1)
                if aug_flag > 0.2:
                    X = affine_aug(X)

                aug_flag = random.uniform(0, 1)
                if aug_flag > 0.2:
                    Y = affine_aug(Y)

                X = F.interpolate(X, size=imgshape, mode='trilinear')
                Y = F.interpolate(Y, size=imgshape, mode='trilinear')

                reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

                # --- Forward Pass with Uncertainty Output ---
                compose_field_e0_lvl1_xy, warpped_inputx_lvl1_out_xy, y_xy, output_disp_e0_v_xy, lvl1_v_xy, lvl2_disp_xy, e0_xy, output_uncertainty_xy = model(X, Y, reg_code)
                compose_field_e0_lvl1_yx, warpped_inputx_lvl1_out_yx, y_yx, output_disp_e0_v_yx, lvl1_v_yx, lvl2_disp_yx, e0_yx, output_uncertainty_yx = model(Y, X, reg_code)
                
                # Apply sigmoid to ensure uncertainty is in [0, infinite]
                output_uncertainty_xy = uncertainty_activation(output_uncertainty_xy)
                output_uncertainty_yx = uncertainty_activation(output_uncertainty_yx)

                # --- Uncertainty-Aware Loss Calculation ---
                # Generate the brain mask based on the fixed image Y
                brain_mask_1ch = (Y > 0).float()
                brain_mask_1ch_yx = (X > 0).float()
                
                # Now, expand the mask to 3 channels to match the uncertainty map
                # brain_mask_3ch = brain_mask_1ch.expand_as(output_uncertainty_xy)
                
                # 1. Similarity Loss (NCC) for X->Y - WITHOUT uncertainty weighting initially
                # Start without uncertainty weighting to let registration stabilize first
                if step < 2000:  # First 2000 steps: train registration without uncertainty
                    loss_multiNCC = loss_similarity(warpped_inputx_lvl1_out_xy, Y, None) + loss_similarity(warpped_inputx_lvl1_out_yx, X, None)
                else:
                    # CORRECTED: Use post-hoc normalization for weighting
                    output_uncertainty = (output_uncertainty_xy + output_uncertainty_yx) / 2
                    masked_M = output_uncertainty * brain_mask_1ch
                    masked_M_yx = output_uncertainty * brain_mask_1ch_yx
                    # Find the maximum M value inside the brain. We use the 95th percentile for robustness.
                    # masked_M = output_uncertainty_xy * brain_mask_1ch
                    # masked_M_yx = output_uncertainty_yx * brain_mask_1ch_yx
                    # masked_M = (masked_M + masked_M_yx) / 2
                    M_99th_percentile = torch.quantile(masked_M.view(masked_M.size(0), -1), 0.99, dim=1, keepdim=True)
                    M_max = M_99th_percentile.view(masked_M.size(0), 1, 1, 1, 1).expand_as(output_uncertainty_xy)

                    # Normalize M to a pseudo-[0,1] range (Uncertainty Map)
                    normalized_M_for_weighting = torch.clamp(output_uncertainty_xy / (M_max + 1e-6), 0.0, 1.0)
                    
                    # alpha = min(1.0, (step - 2000) / 10000.0)
                    # Use normalized M for the certainty map: 1 - normalized_M
                    weighted_certainty_map = 1.0 - masked_M
                    weighted_certainty_map_yx = 1.0 - masked_M_yx
                    loss_multiNCC = loss_similarity(warpped_inputx_lvl1_out_xy, Y, weighted_certainty_map) + loss_similarity(warpped_inputx_lvl1_out_yx, X, weighted_certainty_map_yx)
                    # Gradually introduce uncertainty weighting
                    # uncertainty_mask = torch.nn.Sigmoid(output_uncertainty_xy)
                    # alpha = min(1.0, (step - 2000) / 10000.0)
                    # weighted_certainty_map = 1.0 - alpha * uncertainty_mask * brain_mask_1ch
                    # loss_multiNCC = loss_similarity(warpped_inputx_lvl1_out_xy, Y, weighted_certainty_map)
                '''
                # 2. Generate the SUPERVISION TARGET for uncertainty
                with torch.no_grad():
                    # --- NEW: NO MEDIAN FILTER. USE THE RAW NCC MAP ---
                    local_ncc_map = calculate_local_ncc(warpped_inputx_lvl1_out_xy, Y)
                    target_error_map = 1.0 - local_ncc_map
                    
                    # Use robust normalization - normalize by 95th percentile instead of max
                    masked_error = target_error_map * brain_mask_1ch
                    flattened_error = masked_error.view(masked_error.size(0), -1)
                    
                    # Calculate 95th percentile for each batch item
                    batch_quantiles = torch.quantile(flattened_error, 0.95, dim=1, keepdim=True)
                    batch_quantiles = batch_quantiles.view(masked_error.size(0), 1, 1, 1, 1)
                    
                    # Normalize with clipping to [0,1]
                    normalized_target_uncertainty = torch.clamp(masked_error / (batch_quantiles + 1e-6), 0, 1)
                    
                    # Increase the contrast of the supervision target to focus on high-error regions
                    normalized_target_uncertainty = normalized_target_uncertainty**2
                    
                    normalized_target_uncertainty = normalized_target_uncertainty.expand_as(output_uncertainty_xy)
                '''
                # --- [REWRITTEN] UNCERTAINTY SUPERVISION ---
                # This block uses a high-error mask to focus the supervision on key regions.
                with torch.no_grad():
                    '''
                    # 2. Generate the SUPERVISION TARGET for uncertainty
                    # Calculate local absolute difference as the error proxy
                    target_error_map = torch.abs(warpped_inputx_lvl1_out_xy - Y)
                    
                    # --- NEW: Apply a median filter to remove noise from the target map ---
                    # This will require a filter from a library like kornia or scipy
                    # We'll use a dummy function here
                    target_error_map = median_filter_3d(target_error_map)
                    
                    # Use robust normalization - normalize by 95th percentile instead of max
                    masked_error = target_error_map * brain_mask_1ch
                    flattened_error = masked_error.view(masked_error.size(0), -1)
                    
                    # Calculate 95th percentile for each batch item
                    batch_quantiles = torch.quantile(flattened_error, 0.95, dim=1, keepdim=True)
                    batch_quantiles = batch_quantiles.view(masked_error.size(0), 1, 1, 1, 1)
                    
                    # Normalize with clipping to [0,1]
                    normalized_target_uncertainty = torch.clamp(masked_error / (batch_quantiles + 1e-6), 0, 1)
                    # --- NEW: Increase the contrast of the supervision target to focus on high-error regions ---
                    # This makes the signal from the tumor stronger relative to minor noise.
                    normalized_target_uncertainty = normalized_target_uncertainty**2
                    normalized_target_uncertainty = normalized_target_uncertainty.expand_as(output_uncertainty_xy)
                    '''
                    
                    # 1. Calculate the raw supervision target (error map)
                    # local_ncc_map = calculate_local_ncc(warpped_inputx_lvl1_out_xy, Y)
                    # target_error_map = 1.0 - local_ncc_map
                    target_error_map = torch.abs(warpped_inputx_lvl1_out_xy - Y)
                    target_error_map_yx = torch.abs(warpped_inputx_lvl1_out_yx - X)
                    
                    # 2. Normalize the error map using the 99th percentile, within the brain mask
                    masked_error = target_error_map * brain_mask_1ch
                    masked_error_yx = target_error_map_yx * brain_mask_1ch_yx
                    flattened_error = masked_error.view(masked_error.size(0), -1)
                    batch_quantiles = torch.quantile(flattened_error, 0.99, dim=1, keepdim=True)
                    batch_quantiles = batch_quantiles.view(masked_error.size(0), 1, 1, 1, 1)
                    # normalized_target_uncertainty = torch.clamp(masked_error / (batch_quantiles + 1e-6), 0, 1)
                    
                    # 3. Identify the highest-error regions using the 95th percentile
                    # This creates a sparse binary mask where only the most significant errors are marked
                    # error_quantiles = torch.quantile(normalized_target_uncertainty.view(normalized_target_uncertainty.size(0), -1), 0.95, dim=1, keepdim=True)
                    # error_quantiles = error_quantiles.view(normalized_target_uncertainty.size(0), 1, 1, 1, 1)
                    # high_error_mask = (normalized_target_uncertainty > error_quantiles).float()

                    # 4. Expand the 1-channel mask to 3 channels to match the output uncertainty tensor
                    # high_error_mask_3ch = high_error_mask.expand_as(output_uncertainty_xy)
                    
                    # 5. Apply contrast enhancement (squaring) to the supervision target
                    # This is done AFTER creating the mask to ensure the thresholding is based on the raw normalized values
                    # The new target is now sparse and has enhanced contrast
                    normalized_target_uncertainty = masked_error / (batch_quantiles + 1e-6)
                    # target_for_loss = normalized_target_uncertainty
                    target_for_loss = masked_error
                    target_for_loss_yx = masked_error_yx
                    # masked_target = target_for_loss * high_error_mask_3ch

                # Now, apply your uncertainty loss using the new high-error mask.
                # This loss term will only be calculated on the most critical voxels, ignoring low-error noise.
                # A smooth_l1_loss is used for robust regression.
                # loss_uncertainty_supervision = F.smooth_l1_loss(
                #     output_uncertainty_xy, 
                #     target_for_loss,
                #     reduction='mean'
                # )
                

                # Now, apply the NLL-based uncertainty loss.
                # M is output_uncertainty_xy. ErrorMap is target_for_loss.
                epsilon = 1e-6  # For numerical stability (prevent division by zero or log(0))

                # 1. Ensure the predicted uncertainty (M) is positive and bounded.
                # The original NLL loss assumes M is a variance/std. dev. (must be positive).
                # Using softplus or an activation that forces positivity is usually done on the network output.
                # Since your output is likely already >= 0 from a final activation (e.g., Sigmoid/ReLU), 
                # we enforce positivity with a small offset for stability.
                M_positive = output_uncertainty_xy + epsilon
                M_positive_yx = output_uncertainty_yx + epsilon

                # 2. Calculate the NLL components: (ErrorMap / M) + log(M)
                # M_positive is M(p)
                # target_for_loss is ErrorMap(p)

                # Term 1: ErrorMap / M
                term1 = target_for_loss / M_positive
                term1_yx = target_for_loss_yx / M_positive_yx

                # Term 2: log(M)
                term2 = torch.log(M_positive)
                term2_yx = torch.log(M_positive_yx)

                # Apply the brain mask to the NLL loss
                nll_loss_map = (term1 + term2) * brain_mask_1ch
                nll_loss_map_yx = (term1_yx + term2_yx) * brain_mask_1ch_yx

                # Total Loss: Sum over all voxels inside the brain, then divide by the number of brain voxels.
                # We must use sum and then divide by the total number of non-zero mask pixels for correct mean.
                loss_uncertainty_supervision = torch.sum(nll_loss_map) / torch.sum(brain_mask_1ch) + torch.sum(nll_loss_map_yx) / torch.sum(brain_mask_1ch_yx)
                '''
                # 3. EXPLICIT UNCERTAINTY SUPERVISION LOSS - THIS IS YOUR MOST IMPORTANT TERM!
                # Use a robust loss function that's less sensitive to outliers
                loss_uncertainty_supervision = F.smooth_l1_loss(
                    output_uncertainty_xy * brain_mask_3ch, 
                    normalized_target_uncertainty,
                    reduction='mean'
                )
                '''
                # 4. Uncertainty Regularization Loss - encourage sparsity in uncertainty
                # We use a much smaller weight here to not overwhelm the supervision signal.
                # loss_uncertainty_reg = 0.1 * torch.mean(output_uncertainty_xy)
                # loss_uncertainty_reg = 0.001 * torch.mean(output_uncertainty_xy * brain_mask_1ch)

                # reg2 - use velocity
                _, _, x, y, z = compose_field_e0_lvl1_xy.shape
                norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=compose_field_e0_lvl1_xy.dtype, device=compose_field_e0_lvl1_xy.device)
                norm_vector[0, 0, 0, 0, 0] = z
                norm_vector[0, 1, 0, 0, 0] = y
                norm_vector[0, 2, 0, 0, 0] = x
                loss_regulation = loss_smooth(compose_field_e0_lvl1_xy * norm_vector) + loss_smooth(compose_field_e0_lvl1_yx * norm_vector)

                # --- Combine All Loss Terms ---
                if step > 2000:
                    # Use a larger weight for the supervision loss. This is a critical hyperparameter.
                    lambda_supervision = 0.1
                    loss = (1. - reg_code) * loss_multiNCC + \
                        reg_code * loss_regulation + \
                        lambda_supervision * loss_uncertainty_supervision
                        # loss_uncertainty_reg
                else:
                    loss = (1. - reg_code) * loss_multiNCC + reg_code * loss_regulation
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                        
                # --- Progress output ---
                if step < 2000:
                    sys.stdout.write(
                        "\r" + 'step "{0}" -> total loss "{1:.4f}" - sim "{2:.4f}" - reg "{3:.4f}"'.format(
                            step, loss.item(), loss_multiNCC.item(), loss_regulation.item()))
                else:
                    sys.stdout.write(
                        "\r" + 'step "{0}" -> total loss "{1:.4f}" - sim "{2:.4f}" - reg "{3:.4f}" - unc_sup "{4:.4f}"'.format(
                            step, loss.item(), loss_multiNCC.item(), loss_regulation.item(), 
                            loss_uncertainty_supervision.item()))
                sys.stdout.flush()
                
                # Visualize periodically
                if step % 1000 == 0:
                    print(f"\nUncertainty stats - min: {output_uncertainty_xy.min():.4f}, max: {output_uncertainty_xy.max():.4f}, mean: {output_uncertainty_xy.mean():.4f}")
                    save_visualizations(X, Y, warpped_inputx_lvl1_out_xy, output_uncertainty_xy, f"step_{step}_visualization.png")

            
                # with lr 1e-3 + with bias
                if (step % n_checkpoint == 0):               
                    modelname = model_dir + '/' + str(fold+1) + model_name + "stagelvl3_" + str(step) + '.pth'
                    torch.save(model.state_dict(), modelname)
                    # torch.save(specialist_head.state_dict(), os.path.join(model_dir, f'specialist_head_step_{step}.pth'))
                    np.save(model_dir + '/loss' + str(fold+1) + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

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

                    # assert len(val_fixed_t1ce_list) == len(val_moving_t1ce_list)

                    valid_generator = Data.DataLoader(
                        Validation_Brats(val_fixed, val_moving, val_fixed_csv_list,
                                        val_moving_csv_list, norm=True), batch_size=1,
                        shuffle=False, num_workers=2)

                    use_cuda = True
                    device = torch.device("cuda" if use_cuda else "cpu")
                    # dice_total = []
                    tre_total = []
                    print("\nValiding...")
                    for batch_idx, data in enumerate(valid_generator):
                        # X, Y, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                        #                          data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
                        Y, X, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), \
                                                data['move_label'].numpy()[0], data['fixed_label'].numpy()[0]
                        ori_img_shape = X.shape[2:]
                        h, w, d = ori_img_shape
                        
                        X = F.interpolate(X, size=imgshape, mode='trilinear')
                        Y = F.interpolate(Y, size=imgshape, mode='trilinear')
                        brain_mask_1ch = (Y > 0).float()
                        # Helper function for intensity normalization
                        def normalize_slice(img_slice):
                            """Scales a 2D image slice to the [0, 1] range."""
                            min_val = img_slice.min()
                            max_val = img_slice.max()
                            if max_val - min_val > 1e-6: # Avoid division by zero
                                return (img_slice - min_val) / (max_val - min_val)
                            else:
                                return img_slice - min_val # Return a zero image if it's flat
                                        # x_in = torch.cat((X, Y),dim=1)

                        with torch.no_grad():
                            reg_code = torch.tensor([0.3], dtype=X.dtype, device=X.device).unsqueeze(dim=0)
                            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, output_uncertainty_xy = model(X, Y, reg_code)
                            output_uncertainty_xy = uncertainty_activation(output_uncertainty_xy)
                            # F_X_Y, X_Y, Y_4x = model(x_in)
                            # X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1), grid_unit).data.cpu().numpy()[0, 0, :, :, :]
                            # Y_label = Y_label.data.cpu().numpy()[0, 0, :, :, :]
                            # Find the maximum M value inside the brain. We use the 95th percentile for robustness.
                            masked_M = output_uncertainty_xy * brain_mask_1ch
                            M_99th_percentile = torch.quantile(masked_M.view(masked_M.size(0), -1), 0.99, dim=1, keepdim=True)
                            M_max = M_99th_percentile.view(masked_M.size(0), 1, 1, 1, 1).expand_as(output_uncertainty_xy)

                            # Normalize M to a pseudo-[0,1] range (Uncertainty Map)
                            normalized_M_for_weighting = torch.clamp(output_uncertainty_xy / (M_max + 1e-6), 0.0, 1.0)
                            # M_99th_percentile = torch.quantile(masked_M.view(masked_M.size(0), -1), 0.99, dim=1, keepdim=True)
                            # M_max = M_99th_percentile.view(masked_M.size(0), 1, 1, 1, 1).expand_as(output_uncertainty_xy)

                            # Normalize M to a pseudo-[0,1] range (Uncertainty Map)
                            # normalized_M_for_weighting = torch.clamp(output_uncertainty_xy / (M_max + 1e-6), 0.0, 1.0)
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
                            # print(f"\nSaving visualizations for step {step}...")
                            # save_visualizations(X, Y, X_Y, output_uncertainty_xy, f"registration_vis_{step}.png")
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
                            # Get the 3 uncertainty channels and slice them
                            uncertainty_x_slice = masked_M.squeeze().cpu().detach().numpy()[ :, :, masked_M.shape[-1] // 2]
                            uncertainty_y_slice = normalized_M_for_weighting.squeeze().cpu().detach().numpy()[ :, :, normalized_M_for_weighting.shape[-1] // 2]
                            uncertainty_z_slice = output_uncertainty_xy.squeeze().cpu().detach().numpy()[ :, :, output_uncertainty_xy.shape[-1] // 2]

                            # Filter landmarks that are close to this slice (within ±2 slices)
                            slice_thickness = 80
                            relevant_indices = np.where(
                                (fixed_keypoints[:, 2] >= z_slice - slice_thickness) & 
                                (fixed_keypoints[:, 2] <= z_slice + slice_thickness)
                            )[0]

                            if len(relevant_indices) > 0:
                                fig, axes = plt.subplots(2, 3, figsize=(15, 5))
                                fig.suptitle(f'Step: {step} - Landmark Alignment (Slice {z_slice}) - TRE: {tre_score:.2f}mm', fontsize=14)
                                
                                # Fixed image with fixed landmarks
                                axes[0, 0].imshow(full_fixed_slice, cmap='gray')
                                fixed_slice_landmarks = moving_keypoints[relevant_indices]
                                axes[0, 0].scatter(fixed_slice_landmarks[:, 1], fixed_slice_landmarks[:, 0], 
                                            c='green', s=50, marker='o', label='Fixed', alpha=0.8)
                                axes[0, 0].set_title('Fixed Image + Landmarks')
                                axes[0, 0].legend()
                                    
                                # Moving image with moving landmarks
                                axes[0, 1].imshow(full_moving_slice, cmap='gray')
                                moving_slice_landmarks = fixed_keypoints[relevant_indices]
                                axes[0, 1].scatter(moving_slice_landmarks[:, 1], moving_slice_landmarks[:, 0], 
                                            c='red', s=50, marker='x', label='Moving', alpha=0.8)
                                axes[0, 1].set_title('Moving Image + Landmarks')
                                axes[0, 1].legend()
                                    
                                # Warped image with warped landmarks and fixed landmarks
                                axes[0, 2].imshow(full_warped_slice, cmap='gray')
                                warped_slice_landmarks = warped_moving_keypoint[relevant_indices]
                                    
                                # Plot warped landmarks
                                axes[0, 2].scatter(warped_slice_landmarks[:, 1], warped_slice_landmarks[:, 0], 
                                            c='blue', s=50, marker='s', label='Warped', alpha=0.8)
                                # Plot fixed landmarks for comparison
                                axes[0, 2].scatter(moving_slice_landmarks[:, 1], moving_slice_landmarks[:, 0], 
                                            c='green', s=30, marker='o', label='Fixed', alpha=0.6)
                                    
                                # Draw lines between corresponding points to show displacement
                                for i in range(len(relevant_indices)):
                                    idx = relevant_indices[i]
                                    axes[0, 2].plot([warped_moving_keypoint[idx, 1], fixed_keypoints[idx, 1]],
                                                [warped_moving_keypoint[idx, 0], fixed_keypoints[idx, 0]],
                                                'y-', linewidth=1, alpha=0.6)

                                axes[0, 2].set_title('Warped + Fixed Landmarks (Yellow lines show error)')
                                axes[0, 2].legend()
                                
                                # Plot the uncertainty maps for each dimension
                                axes[1, 0].imshow(uncertainty_x_slice, cmap='gray')
                                axes[1, 0].set_title('Uncertainty (X-axis)')
                                axes[1, 0].axis('off')

                                axes[1, 1].imshow(uncertainty_y_slice, cmap='gray')
                                axes[1, 1].set_title('Uncertainty (Y-axis)')
                                axes[1, 1].axis('off')

                                axes[1, 2].imshow(uncertainty_z_slice, cmap='gray')
                                axes[1, 2].set_title('Uncertainty (Z-axis)')
                                axes[1, 2].axis('off')
                                    
                                plt.tight_layout()
                                plt.savefig(f"{model_dir}/landmarks_{fold+1}_step_{step}_batch_{batch_idx}_slice_{z_slice}.png", 
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
                                plt.savefig(f"{model_dir}/landmark_{fold+1}_errors_step_{step}_batch_{batch_idx}.png",
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
                        # --- End of Visualization Code ---

                    tre_total = np.array(tre_total)
                    print("TRE mean: ", tre_total.mean())
                    log_dir = f"/workspace/DIRAC/Log/{fold + 1}Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_.txt"
                    with open(log_dir, "a") as log:
                        log.write(str(step) + ":" + str(tre_total.mean()) + "\n")

                    # if step == freeze_step:
                    #     model.unfreeze_modellvl2()
                    #     # num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    #     # print("\nmodel_lvl2_num_param_requires_grad: ", num_param)

                step += 1

                if step > iteration_lvl3:
                    break
            print("one epoch pass")
        np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)
            


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
    if not os.path.isdir("/workspace/DIRAC/Log"):
        os.mkdir("/workspace/DIRAC/Log")

    log_dir = "/workspace/DIRAC/Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation TRE log for " + model_name[0:-1] + ":\n")

    img_h, img_w, img_d = 160, 160, 80
    imgshape = (img_h, img_w, img_d)
    imgshape_4 = (img_h // 4, img_w // 4, img_d // 4)
    imgshape_2 = (img_h // 2, img_w // 2, img_d // 2)

    range_flow = 0.4
    print("Training %s ..." % model_name)
    train()