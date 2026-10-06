import torch
import json
import matplotlib.pyplot as plt

from fit_3d import fit_3D, evaluate_3D
from image import load_normalized_image, save_normalized_image
from fit import get_device

ROOT_PATH = "train_images/spheres/"
device = get_device()
N = 4000

def convert_frames(frames, H, W):
    N_frames = len(frames)
    # (N_frames, 3, 3)
    camera_R = torch.empty((N_frames, 3, 3), device=device) 
    # (N_frames, 3)
    camera_t = torch.empty((N_frames, 3), device=device) 
    images = torch.empty((N_frames, H, W, 3), dtype=torch.float32, device=device)
    # Normal python array, mainly for debugging purposes and ease of recognizing images
    image_names = []
    cnt = 0
    for frame in frames:
        img_path = frame['file']
        img = load_normalized_image(ROOT_PATH + img_path)
        img = img.to(device)
        image_names.append(img_path)

        R_wc = torch.tensor(frame['R_wc'], device=device)
        t = torch.tensor(frame['t'], device=device)

        camera_R[cnt] = R_wc
        camera_t[cnt] = t
        images[cnt] = img
        cnt += 1
    return (camera_R, camera_t, images, image_names)

def load_cameras(root_path):
    cameras_path = root_path + "cameras.json"
    with open(cameras_path, "r") as f:
        cameras_data = json.load(f)
        # Shared across all
        W = cameras_data['width']
        H = cameras_data['height']
        K = torch.tensor(cameras_data['K'], device=device)

        train_frames = cameras_data['frames']
        train_R, train_t, train_img, train_img_names = convert_frames(train_frames, H, W)

        val_frames = cameras_data['val_frames']
        val_R, val_t, val_img, val_img_names = convert_frames(val_frames, H, W)

    return ((train_R, train_t, train_img, H, W, K), train_img_names,
            (val_R, val_t, val_img, H, W, K), val_img_names)

def train_3D():
    # set seed so can compare consistently
    torch.manual_seed(0)
    train_cameras, train_names, val_cameras, val_names = load_cameras(ROOT_PATH)
    mu3, log_s, quat, color, op_raw = fit_3D(train_cameras)

    N_g = mu3.shape[0]
    print(f"\033[31mNum Gaussians: {N_g}\033[0m")

    train_R, train_t, train_img, H, W, K = train_cameras
    val_R, val_t, val_img, _, _, _ = val_cameras
    num_train_cams = len(train_img)
    num_val_cams = len(val_img)
    train_PSNR_avg = 0.0
    for i in range(num_train_cams):
        cam = (train_R[i], train_t[i], train_img[i], H, W, K)
        name = train_names[i]
        img, psnr = evaluate_3D(mu3, log_s, quat, color, op_raw, cam)
        save_normalized_image(ROOT_PATH + "results/" + name, img)
        train_PSNR_avg += psnr / (num_train_cams)
        print(f"\033[31mPSNR{psnr} for Image: {name} \033[0m")
    print(f"Average Train PSNR: {train_PSNR_avg}")

    val_PSNR_avg = 0.0
    for i in range(num_val_cams):
        cam = (val_R[i], val_t[i], val_img[i], H, W, K)
        name = val_names[i]
        img, psnr = evaluate_3D(mu3, log_s, quat, color, op_raw, cam)
        save_normalized_image(ROOT_PATH + "results/" + name, img)
        val_PSNR_avg += psnr / (num_val_cams)
        print(f"\033[31mPSNR{psnr} for Image: {name} \033[0m")
    print(f"Average Val PSNR: {val_PSNR_avg}")
    

train_3D()
