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
import matplotlib.pyplot as plt

from Functions import Dataset_bratsreg_bidirection, Validation_Brats, \
    generate_grid_unit
from bratsreg_model_stage import Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl1, \
    Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl2, Miccai2021_LDR_laplacian_unit_disp_add_AdaIn_lvl3, Miccai2021_LDR_laplacian_TransMorph_lvl3, \
    SpatialTransform_unit, smoothloss, multi_resolution_NCC_weight, multi_resolution_NCC_weight_2D, MultiResolution_Cosine, DINO_Cosine_Similarity, DINO_Cosine_Loss, SpecialistHead
from cnn_swin import Dual_FusionMorph

from transformers import AutoImageProcessor, AutoModel

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
    specialist_head = SpecialistHead().cuda()
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
    fixed_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
    moving_t1ce_list = sorted(glob.glob(f"{datapath}/BraTSReg_*/*_t1ce.nii.gz"))
    moving_t1ce_list = sorted([path for path in moving_t1ce_list if path not in fixed_t1ce_list])

    # # LPBA
    # names = sorted(glob.glob(datapath + '/S*_norm.nii'))[0:30]

    # grid = generate_grid(imgshape)
    # grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    grid_unit = generate_grid_unit(imgshape)
    grid_unit = torch.from_numpy(np.reshape(grid_unit, (1,) + grid_unit.shape)).cuda().float()

    optimizer = torch.optim.Adam(list(model.parameters()) + list(specialist_head.parameters()), lr=lr)
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
        model_path = "/workspace/DIRAC/Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_stagelvl3_72000.pth"
        print("Loading weight: ", model_path)
        step = 72000
        model.load_state_dict(torch.load(model_path))
        specialist_head.load_state_dict(torch.load(os.path.join(model_dir, f'specialist_head_step_{step}.pth')))
        temp_lossall = np.load("/workspace/DIRAC/Model/Brats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github/lossBrats_NCC_disp_fea6b5_AdaIn64_t1ce_fbcon_occ01_inv1_a0015_aug_mean_fffixed_github_stagelvl3_72000.npy")
        lossall[:, 0:72000] = temp_lossall[:, 0:72000]

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
            # x_in_xy = torch.cat((X, Y), dim=1)
            # x_in_yx = torch.cat((Y,X), dim=1)
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)

            # compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0
            # lap
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, predicted_stress_xy, predicted_mask_xy = model(X, Y, reg_code)
            # F_X_Y, X_Y, Y_4x = model(x_in_xy)
            # F_Y_X, Y_X, X_4x = model(x_in_yx)
            F_Y_X, Y_X, X_4x, F_yx, F_yx_lvl1, F_yx_lvl2, _, predicted_stress_yx, predicted_mask_yx = model(Y, X, reg_code)
            
            # F_X_Y, X_Y, Y_4x = model(X, Y)
            # F_Y_X, Y_X, X_4X = model(Y, X)

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
            # Get the spatial dimensions from the input tensor
            D, H, W = X.shape[2:]  # X has shape [batch, channels, depth, height, width]
            occ_xy = (smo_norm_diff_fw > thresh_fw).float()  # y mask
            occ_yx = (smo_norm_diff_bw > thresh_bw).float()  # x mask

            occ_xy_l = F.relu(smo_norm_diff_fw - thresh_fw) * 10.
            occ_yx_l = F.relu(smo_norm_diff_bw - thresh_bw) * 10.

            # mask occ
            occ_xy = occ_xy * fw_mask
            occ_yx = occ_yx * bw_mask

            # --- YOUR NEW CODE STARTS HERE ---
            # Dilate the occlusion mask to get the Physics ROI
            # --- YOUR NEW CODE STARTS HERE ---
            # Dilate the occlusion mask to get the Physics ROI
            kernel_size = 5
            dilated_occ_xy = F.max_pool3d(occ_xy, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            dilated_occ_yx = F.max_pool3d(occ_yx, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
            physics_roi_xy = dilated_occ_xy
            physics_roi_yx = dilated_occ_yx

            # Ensure we only consider points inside the brain
            # physics_roi_xy = physics_roi_xy * fw_mask
            # physics_roi_yx = physics_roi_yx * bw_mask

            stress_field_xy = compute_stress_field_nonlinear(F_X_Y)
            stress_field_yx = compute_stress_field_nonlinear(F_Y_X)
            
            # physics_loss_xy = calculate_physics_loss(F_X_Y, physics_roi_xy)
            # physics_loss_yx = calculate_physics_loss(F_Y_X, physics_roi_yx)
            physics_loss_xy = calculate_equilibrium_loss(predicted_stress_xy, physics_roi_xy)
            physics_loss_yx = calculate_equilibrium_loss(predicted_stress_yx, physics_roi_yx)
            physics_loss = physics_loss_xy + physics_loss_yx
            # Physics loss is the L2 norm of the stress divergence
            # physics_loss = torch.mean(torch.norm(divergence_samples, dim=1) ** 2)
            '''
            # Plotting (using matplotlib)
            import matplotlib.pyplot as plt
            # Get a middle slice for visualization
            slice_idx = X.shape[2] // 2  # Middle slice in depth dimension
            
            # Extract slices for visualization
            X_slice = X[0, 0, slice_idx].cpu().detach().numpy()  # Fixed image
            Y_slice = Y[0, 0, slice_idx].cpu().detach().numpy()  # Moving image
            X_Y_slice = X_Y[0, 0, slice_idx].cpu().detach().numpy()  # Warped image
            Y_X_slice = Y_X[0, 0, slice_idx].cpu().detach().numpy()  # Warped image
            
            # Extract mask slices
            occ_xy_slice = occ_xy[0, 0, slice_idx].cpu().detach().numpy()  # Occlusion mask XY
            occ_yx_slice = occ_yx[0, 0, slice_idx].cpu().detach().numpy()  # Occlusion mask YX
            physics_roi_xy_slice = physics_roi_xy[0, 0, slice_idx].cpu().detach().numpy()  # Physics ROI XY
            physics_roi_yx_slice = physics_roi_yx[0, 0, slice_idx].cpu().detach().numpy()  # Physics ROI YX
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
            # Row 1: Images
            axes[0, 0].imshow(X_slice, cmap='gray')
            axes[0, 0].set_title('Fixed Image (X)')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(Y_slice, cmap='gray')
            axes[0, 1].set_title('Moving Image (Y)')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(X_Y_slice, cmap='gray')
            axes[0, 2].set_title('Warped X→Y')
            axes[0, 2].axis('off')
            
            axes[0, 3].imshow(Y_X_slice, cmap='gray')
            axes[0, 3].set_title('Warped Y→X')
            axes[0, 3].axis('off')
            
            # Row 2: Masks with overlays
            axes[1, 0].imshow(X_slice, cmap='gray')
            axes[1, 0].imshow(occ_xy_slice, cmap='Reds', alpha=0.5)
            axes[1, 0].set_title('Occlusion Mask XY (Red)')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(Y_slice, cmap='gray')
            axes[1, 1].imshow(occ_yx_slice, cmap='Reds', alpha=0.5)
            axes[1, 1].set_title('Occlusion Mask YX (Red)')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(X_slice, cmap='gray')
            axes[1, 2].imshow(physics_roi_xy_slice, cmap='Blues', alpha=0.5)
            axes[1, 2].set_title('Physics ROI XY (Blue)')
            axes[1, 2].axis('off')
            
            axes[1, 3].imshow(Y_slice, cmap='gray')
            axes[1, 3].imshow(physics_roi_yx_slice, cmap='Blues', alpha=0.5)
            axes[1, 3].set_title('Physics ROI YX (Blue)')
            axes[1, 3].axis('off')
            
            plt.savefig(f'physics_roi_visualization_step_{step}.png')
            plt.close(fig)
            '''
            
            mask_xy_cal = 1. - occ_xy
            mask_yx_cal = 1. - occ_yx
            mask_xy = 1. - predicted_mask_xy
            mask_yx = 1. - predicted_mask_yx
            
            # Assuming you have defined your weighting coefficients
            # alpha: High (e.g., 1.0) for core healthy tissue
            # beta: Medium (e.g., 0.1 to 0.5) for the boundary
            # gamma: Zero (e.g., 0.0) for the missing area
            alpha, beta, gamma = 0.7, 1.0, 0.3

            # --- START: CORRECTED PERCEPTUAL LOSS CALCULATION (WITH 3-PART MASKING) ---
            image_pairs = [(X_Y, Y_4x, occ_xy, fw_mask), (Y_X, X_4x, occ_yx, bw_mask)]
            total_perceptual_loss = torch.tensor(0.0, device=X.device)

            for warped_img, fixed_img, occ_mask, brain_mask in image_pairs:
                # --- 1. Select Axis and Permute Tensors (No changes here) ---
                axis = random.choice([0, 1, 2])
                if axis == 0:
                    dim_size = warped_img.shape[2]
                    permute_dims = (0, 2, 1, 3, 4)
                    target_h, target_w = warped_img.shape[3], warped_img.shape[4]
                elif axis == 1:
                    dim_size = warped_img.shape[3]
                    permute_dims = (0, 3, 1, 2, 4)
                    target_h, target_w = warped_img.shape[2], warped_img.shape[4]
                else:
                    dim_size = warped_img.shape[4]
                    permute_dims = (0, 4, 1, 2, 3)
                    target_h, target_w = warped_img.shape[2], warped_img.shape[3]
                
                warped_permuted = warped_img.permute(*permute_dims)
                fixed_permuted = fixed_img.permute(*permute_dims)
                mask_permuted = occ_mask.permute(*permute_dims)
                brain_mask_permuted = brain_mask.permute(*permute_dims)

                # --- 2. Identify Valid Slices and Sample ---
                slice_content_sum = brain_mask_permuted.sum(dim=[2, 3, 4])
                valid_slice_indices = torch.nonzero(slice_content_sum > 100, as_tuple=False).squeeze()
                
                if valid_slice_indices.numel() == 0:
                    continue
                if valid_slice_indices.dim() == 2:
                    valid_slice_indices = valid_slice_indices[:, 1]

                num_to_sample = min(num_slices_for_loss, valid_slice_indices.numel())
                shuffled_valid_indices = valid_slice_indices[torch.randperm(valid_slice_indices.numel())]
                slice_indices_to_use = shuffled_valid_indices[:num_to_sample].cuda()
                
                warped_slices = torch.index_select(warped_permuted, 1, slice_indices_to_use)
                fixed_slices = torch.index_select(fixed_permuted, 1, slice_indices_to_use)
                mask_slices = torch.index_select(mask_permuted, 1, slice_indices_to_use)
                
                warped_slices = warped_slices.reshape(-1, warped_slices.shape[2], target_h, target_w)
                fixed_slices = fixed_slices.reshape(-1, fixed_slices.shape[2], target_h, target_w)
                mask_slices = mask_slices.reshape(-1, mask_slices.shape[2], target_h, target_w)
                
                # Repeat for RGB as DINO expects 3 channels
                warped_slices_rgb = warped_slices.repeat(1, 3, 1, 1)
                fixed_slices_rgb = fixed_slices.repeat(1, 3, 1, 1)
                
                # --- 3. Feature Extraction (No changes here) ---
                inputs_warped = dino_processor(images=warped_slices_rgb, return_tensors="pt", do_rescale=False).to('cuda')
                inputs_fixed = dino_processor(images=fixed_slices_rgb, return_tensors="pt", do_rescale=False).to('cuda')
                
                with torch.no_grad():
                    features_warped_raw = dino_model(**inputs_warped, output_hidden_states=True).hidden_states[-4]
                    features_fixed_raw = dino_model(**inputs_fixed, output_hidden_states=True).hidden_states[-4]
                
                patch_features_warped = features_warped_raw[:, 1:, :]
                patch_features_fixed = features_fixed_raw[:, 1:, :]
                
                num_patches = patch_features_warped.shape[1]
                H_feat = W_feat = int(math.sqrt(num_patches))
                
                patch_features_warped = patch_features_warped.permute(0, 2, 1)
                patch_features_fixed = patch_features_fixed.permute(0, 2, 1)
                I_for_loss = patch_features_warped.reshape(patch_features_warped.shape[0], patch_features_warped.shape[1], H_feat, W_feat)
                J_for_loss = patch_features_fixed.reshape(patch_features_fixed.shape[0], patch_features_fixed.shape[1], H_feat, W_feat)

                # --- 4. Create the THREE-PART Masks in the Feature Space ---
                correct_feature_map_size = (H_feat, W_feat)
                mask_downsampled = F.interpolate(mask_slices.float(), size=correct_feature_map_size, mode='nearest')
                
                # Dilate the original mask
                kernel_size = 3
                dilated_mask_downsampled = F.max_pool2d(mask_downsampled, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                
                # Define the three distinct masks
                core_healthy_mask = 1.0 - dilated_mask_downsampled
                boundary_mask = dilated_mask_downsampled - mask_downsampled
                missing_area_mask = mask_downsampled

                # --- 5. Calculate the Three Loss Terms and Combine ---
                loss_healthy = torch.tensor(0.0, device=X.device)
                if core_healthy_mask.sum() > 0:
                    loss_healthy = loss_correspondence(I_for_loss, J_for_loss, core_healthy_mask)
                
                loss_boundary = torch.tensor(0.0, device=X.device)
                if boundary_mask.sum() > 0:
                    loss_boundary = loss_correspondence(I_for_loss, J_for_loss, boundary_mask)

                loss_missing_area = torch.tensor(0.0, device=X.device)
                if missing_area_mask.sum() > 0:
                    # For the missing area, we want to maximize dissimilarity (e.g., use your original dissimilarity logic)
                    loss_missing_area = loss_correspondence(I_for_loss, J_for_loss, missing_area_mask)

                # --- Combine the losses with weights ---
                total_perceptual_loss += (alpha * loss_healthy) + (beta * loss_boundary) + (gamma * loss_missing_area)
                
            # Now use total_perceptual_loss in your final objective function.

            # 6. Combine the losses
            perceptual_loss = total_perceptual_loss
                # perceptual_loss += loss_perceptual(I_for_loss, J_for_loss)
            # --- END: CORRECTED PERCEPTUAL LOSS CALCULATION ---
            perceptual_loss = perceptual_loss / 2
            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x, mask_xy) + loss_similarity(Y_X, X_4x, mask_yx)

            loss_inverse = torch.mean(norm_diff_fw * mask_xy_cal) + torch.mean(norm_diff_bw * mask_yx_cal)
            loss_occ = torch.mean(occ_xy_l) + torch.mean(occ_yx_l)
            loss_constitutive = torch.mean((predicted_stress_xy - stress_field_xy)**2) + \
                    torch.mean((predicted_stress_yx - stress_field_yx)**2)
            loss_mask = torch.mean((predicted_mask_xy - occ_xy)**2) + \
                    torch.mean((predicted_mask_yx - occ_yx)**2)

            # F_X_Y_norm = transform_unit_flow_to_flow_cuda(F_X_Y.permute(0,2,3,4,1).clone())
            # loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_X_Y.shape
            norm_vector = torch.zeros((1, 3, 1, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0, 0] = z
            norm_vector[0, 1, 0, 0, 0] = y
            norm_vector[0, 2, 0, 0, 0] = x
            loss_regulation = loss_smooth(F_X_Y * norm_vector) + loss_smooth(F_Y_X * norm_vector)

            loss = (1. - reg_code) * loss_multiNCC + reg_code * loss_regulation + reg_code * physics_loss + 0.5 * loss_constitutive + 0.5 * loss_mask + (1. - reg_code) * perceptual_loss + occ * loss_occ + inv_con * loss_inverse
            # loss = loss_multiNCC + occ * loss_occ + inv_con * loss_inverse + physics_loss
            
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_inverse.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" -sim_NCC "{2:4f}" -inv "{3:.4f}" -occ "{4:4f}" -smo "{5:.4f} -reg_c "{6:.4f}" - phy "{7:4f}" - stress "{8:4f}" - mask "{9:4f}" -percep "{10:4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_inverse.item(), loss_occ.item(),
                    loss_regulation.item(), reg_code[0].item(), physics_loss.item(), loss_constitutive.item(), loss_mask.item(), perceptual_loss.item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if (step % n_checkpoint == 0):                
                # Visualize a few feature channels for both warped and fixed
                fig, axes = plt.subplots(4, 8, figsize=(20, 10))
                
                # Select random feature channels to visualize
                feature_channels = torch.randperm(768)[:8]  # Random 8 channels
                
                for i, channel_idx in enumerate(feature_channels):
                    # Warped features
                    warped_feat_slice = I_for_loss[0, channel_idx].cpu().detach().numpy()
                    axes[0, i].imshow(warped_feat_slice, cmap='viridis')
                    axes[0, i].set_title(f'Warped Ch{channel_idx}')
                    axes[0, i].axis('off')
                    
                    # Fixed features
                    fixed_feat_slice = J_for_loss[0, channel_idx].cpu().detach().numpy()
                    axes[1, i].imshow(fixed_feat_slice, cmap='viridis')
                    axes[1, i].set_title(f'Fixed Ch{channel_idx}')
                    axes[1, i].axis('off')
                    
                    # Difference
                    diff_feat = warped_feat_slice - fixed_feat_slice
                    axes[2, i].imshow(diff_feat, cmap='RdBu_r', vmin=-3, vmax=3)
                    axes[2, i].set_title(f'Diff Ch{channel_idx}')
                    axes[2, i].axis('off')
                    
                    # Mask (downsampled)
                    mask_slice = mask_slices[0, 0].cpu().detach().numpy()
                    axes[3, i].imshow(mask_slice, cmap='gray')
                    axes[3, i].set_title(f'Mask')
                    axes[3, i].axis('off')
                
                plt.suptitle(f'Step {step} - DINO Feature Visualization\n(8 random channels from 768)', fontsize=16)
                plt.tight_layout()
                plt.savefig(f'{model_dir}/dino_features_step_{step}.png', dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                print(f"Saved DINO feature visualization for step {step}")
                
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                torch.save(specialist_head.state_dict(), os.path.join(model_dir, f'specialist_head_step_{step}.pth'))
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Validation
                val_datapath = '/workspace/DIRAC/Data/BraTSReg/BraTSReg_validation'
                start, end = 0, 20
                val_fixed_csv_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_landmarks.csv"))
                val_moving_csv_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_landmarks.csv"))
                val_moving_csv_list = sorted([path for path in val_moving_csv_list if path not in val_fixed_csv_list])

                val_fixed_t1ce_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_0000_t1ce.nii.gz"))
                val_moving_t1ce_list = sorted(glob.glob(f"{val_datapath}/BraTSReg_*/*_t1ce.nii.gz"))
                val_moving_t1ce_list = sorted(
                    [path for path in val_moving_t1ce_list if path not in val_fixed_t1ce_list])

                # assert len(val_fixed_t1ce_list) == len(val_moving_t1ce_list)

                valid_generator = Data.DataLoader(
                    Validation_Brats(val_fixed_t1ce_list, val_moving_t1ce_list, val_fixed_csv_list,
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
                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, predicted_mask, predicted_stress = model(X, Y, reg_code)
                        # F_X_Y, X_Y, Y_4x = model(x_in)
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
                            plt.savefig(f"{model_dir}/landmarks_step_{step}_batch_{batch_idx}_slice_{z_slice}.png", 
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
                            plt.savefig(f"{model_dir}/landmark_errors_step_{step}_batch_{batch_idx}.png",
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
