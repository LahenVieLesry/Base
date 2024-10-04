import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM, SmoothGradCAMpp
from torchcam.utils import overlay_mask
from param import param
from PIL import Image

device = torch.device(f'cuda:{param.GPU_num}' if torch.cuda.is_available() else 'cpu')

benign_model_name = 'models/train/resnet34_epochs_200_batchsize_128_Adam_lr_0.02_mom_0.9_id_0.pth'
# benign_model_name = 'models/train/000.pth'

benign_tensor = np.load('datasets/train/on/190821dc_nohash_2.npy')
poi_tensor = np.load('datasets/temp/train_wav_layer4/on_190821dc_nohash_2.npy')

# Ensure the tensors have the correct shape (1 channel)
benign_tensor = benign_tensor[:, np.newaxis, :, :]
poi_tensor = poi_tensor[:, np.newaxis, :, :]

tensor_data_1 = torch.tensor(benign_tensor).to(device).float()
tensor_data_2 = torch.tensor(poi_tensor).to(device).float()

def load_model(path):
    model = torch.load(path, map_location=device)
    model.to(device)
    return model

model = load_model(benign_model_name).eval()

# Initialize CAM methods
gradcam_extractor = GradCAM(model, target_layer='layer4')
smoothgradcampp_extractor = SmoothGradCAMpp(model, target_layer='layer4')

# Generate CAM for tensor_data_1
out_1 = model(tensor_data_1)
activation_map_gradcam_1 = gradcam_extractor(out_1.squeeze(0).argmax().item(), out_1)[0]
activation_map_smoothgradcampp_1 = smoothgradcampp_extractor(out_1.squeeze(0).argmax().item(), out_1)[0]

# Generate CAM for tensor_data_2
out_2 = model(tensor_data_2)
activation_map_gradcam_2 = gradcam_extractor(out_2.squeeze(0).argmax().item(), out_2)[0]
activation_map_smoothgradcampp_2 = smoothgradcampp_extractor(out_2.squeeze(0).argmax().item(), out_2)[0]

def visualize_cam(tensor_data_1, activation_map_gradcam_1, activation_map_smoothgradcampp_1, tensor_data_2, activation_map_gradcam_2, activation_map_smoothgradcampp_2, title):
    tensor_data_1 = tensor_data_1.cpu().numpy().squeeze()
    activation_map_gradcam_1 = activation_map_gradcam_1.cpu().numpy().squeeze()
    activation_map_smoothgradcampp_1 = activation_map_smoothgradcampp_1.cpu().numpy().squeeze()
    tensor_data_2 = tensor_data_2.cpu().numpy().squeeze()
    activation_map_gradcam_2 = activation_map_gradcam_2.cpu().numpy().squeeze()
    activation_map_smoothgradcampp_2 = activation_map_smoothgradcampp_2.cpu().numpy().squeeze()

    # Convert numpy arrays to PIL images
    img_1 = Image.fromarray((tensor_data_1 * 255).astype(np.uint8)).convert("RGB")
    img_2 = Image.fromarray((tensor_data_2 * 255).astype(np.uint8)).convert("RGB")

    # Overlay activation maps on the original images
    overlay_img_gradcam_1 = overlay_mask(img_1, Image.fromarray((activation_map_gradcam_1 * 255).astype(np.uint8), mode='L'), alpha=0.5)
    overlay_img_smoothgradcampp_1 = overlay_mask(img_1, Image.fromarray((activation_map_smoothgradcampp_1 * 255).astype(np.uint8), mode='L'), alpha=0.5)
    overlay_img_gradcam_2 = overlay_mask(img_2, Image.fromarray((activation_map_gradcam_2 * 255).astype(np.uint8), mode='L'), alpha=0.5)
    overlay_img_smoothgradcampp_2 = overlay_mask(img_2, Image.fromarray((activation_map_smoothgradcampp_2 * 255).astype(np.uint8), mode='L'), alpha=0.5)

    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    ax[0, 0].imshow(img_1, cmap='gray')
    ax[0, 0].set_title('Original Image 1')
    ax[0, 0].invert_yaxis()
    
    ax[0, 1].imshow(overlay_img_gradcam_1)
    ax[0, 1].set_title('GradCAM 1')
    ax[0, 1].invert_yaxis()
    
    ax[0, 2].imshow(overlay_img_smoothgradcampp_1)
    ax[0, 2].set_title('SmoothGradCAMpp 1')
    ax[0, 2].invert_yaxis()
    
    ax[1, 0].imshow(img_2, cmap='gray')
    ax[1, 0].set_title('Original Image 2')
    ax[1, 0].invert_yaxis()
    
    ax[1, 1].imshow(overlay_img_gradcam_2)
    ax[1, 1].set_title('GradCAM 2')
    ax[1, 1].invert_yaxis()
    
    ax[1, 2].imshow(overlay_img_smoothgradcampp_2)
    ax[1, 2].set_title('SmoothGradCAMpp 2')
    ax[1, 2].invert_yaxis()
    
    plt.suptitle(title)
    plt.savefig(f'gradcam_{title}.png')
    plt.show()

# Visualize CAM results
visualize_cam(tensor_data_1, activation_map_gradcam_1, activation_map_smoothgradcampp_1, tensor_data_2, activation_map_gradcam_2, activation_map_smoothgradcampp_2, 'Tensor Data Comparison')