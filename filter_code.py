import os
import numpy as np
import nibabel as nib
import imageio.v2 as imageio
from skimage.restoration import denoise_nl_means, estimate_sigma
from scipy.ndimage import gaussian_filter, distance_transform_edt, convolve
import cv2

def create_stronger_weight_map(image, sigma=10, threshold_percentile=85):
    threshold = np.percentile(image, threshold_percentile)
    mask = image > threshold
    distance_map = distance_transform_edt(~mask)
    weight_map = gaussian_filter(distance_map, sigma=sigma)
    weight_map = 1 - (weight_map / np.max(weight_map))
    return np.clip(weight_map, 0.5, 1)

def apply_enhanced_nlm_filter(image, weight_map, patch_size=5, patch_distance=6, h_factor=2.5):
    sigma_est = np.mean(estimate_sigma(image))
    nlm_filtered = denoise_nl_means(
        image,
        h=h_factor * sigma_est,
        patch_size=patch_size,
        patch_distance=patch_distance,
        fast_mode=True
    )
    return weight_map * nlm_filtered + (1 - weight_map) * image

def apply_enhanced_anisotropic_diffusion(image, weight_map, num_iter=15, kappa=25, gamma=0.1):
    image = image.astype(np.float32)
    for _ in range(num_iter):
        grad_n = np.roll(image, -1, axis=0) - image
        grad_s = np.roll(image, 1, axis=0) - image
        grad_e = np.roll(image, -1, axis=1) - image
        grad_w = np.roll(image, 1, axis=1) - image
        c_n = np.exp(-(grad_n / kappa) ** 2)
        c_s = np.exp(-(grad_s / kappa) ** 2)
        c_e = np.exp(-(grad_e / kappa) ** 2)
        c_w = np.exp(-(grad_w / kappa) ** 2)
        diffused = image + gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        image = weight_map * diffused + (1 - weight_map) * image
    return np.clip(image, 0, 1)

def darken_main_structure(image, weight_map, darken_factor=0.7):
    inverted_weight_map = 1 - (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map))
    darkened_image = image * (1 - (inverted_weight_map * (1 - darken_factor)))
    return np.clip(darkened_image, 0, 1)

def apply_sharpening(image, alpha=1.5):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_image = convolve(image, kernel)
    return np.clip(image + alpha * (sharpened_image - image), 0, 1)

def preprocess_echocardiography(input_path, output_folder, file_name):
    print(f"Processing {input_path}...")
    nii = nib.load(input_path)
    image_data = nii.get_fdata()
    
    if len(image_data.shape) != 2:  # Skip 3D images
        print(f"Skipping {input_path}, as it is not a 2D image.")
        return
    
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
    weight_map = create_stronger_weight_map(image_data, sigma=12, threshold_percentile=90)
    nlm_filtered = apply_enhanced_nlm_filter(image_data, weight_map)
    smoothed = apply_enhanced_anisotropic_diffusion(nlm_filtered, weight_map)
    darkened_image = darken_main_structure(smoothed, weight_map)
    sharpened_image = apply_sharpening(darkened_image)
    
    resized_image = cv2.resize(sharpened_image, (416, 480))
    
    output_path = os.path.join(output_folder, file_name.replace(".nii.gz", ".png"))
    imageio.imwrite(output_path, (resized_image * 255).astype(np.uint8))
    print(f"Saved: {output_path}")

def process_multiple_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    input_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.nii.gz')])
    for file_name in input_files:
        input_path = os.path.join(input_folder, file_name)
        preprocess_echocardiography(input_path, output_folder, file_name)

input_folder = r"E:\PROJECTS\Mini_Project\sem_6\rough\Operational_database\normal_images"
output_folder = r"E:\PROJECTS\Mini_Project\sem_6\img_filtered_480_416"
process_multiple_files(input_folder, output_folder)
