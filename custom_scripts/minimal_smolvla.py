#!/usr/bin/env python

import torch
import lerobot
try:
    from lerobot.src.lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
except:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.policies.normalize import Normalize, Unnormalize
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess a single image for robot inference"""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Define transforms to match training preprocessing
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),  # Converts to [0, 1] range and CHW format
        # Add normalization if your training used it
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply transforms
    image_tensor = transform(image)  # Shape: (3, 256, 256)
    return image_tensor

def create_minimal_batch(image_path, image2_path, image3_path, device='cuda:0'):
    """Create a minimal batch structure for robot inference"""

    # Load and preprocess images
    image = load_and_preprocess_image(image_path)
    image2 = load_and_preprocess_image(image2_path)
    image3 = load_and_preprocess_image(image3_path)

    # Add batch dimension and move to device
    tmp_ = {
            'image': image.unsqueeze(0).to(device),  # Shape: (1, 3, 256, 256)
            'image2': image2.unsqueeze(0).to(device),  # Shape: (1, 3, 256, 256)
            'image3': image3.unsqueeze(0).to(device),  # Shape: (1, 3, 256, 256)
        }
    batch = {
        **{'observation.'+t : tmp_[t] for t in tmp_},
        'task': ["pick up the object"],
        'observation.state': torch.ones(6).unsqueeze(0).to(device),
    }

    return batch


def inference_with_policy(policy, batch):
    """Run inference with your trained policy"""

    # Set policy to evaluation mode
    policy.eval()

    # Get device from policy
    device = next(policy.parameters()).device

    # Move batch to same device as policy if needed
    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        else:
            return obj

    batch = move_to_device(batch, device)

    # Run inference
    with torch.no_grad():
        # For inference, we typically want just the action prediction
        # The exact method depends on your policy implementation
        if hasattr(policy, 'select_action'):
            # print("batch")
            # print(batch)
            action = policy.select_action(batch)
        else:
            # Fallback to forward pass
            output = policy(batch)
            if isinstance(output, dict) and 'action' in output:
                action = output['action']
            else:
                action = output

    return action


def initialize_smolvla_normalization_stats(policy):
    """Initialize normalization statistics for SmolVLA base model"""

    print("Initializing normalization stats for SmolVLA...")

    for name, module in policy.named_modules():
        if isinstance(module, (Unnormalize, Normalize)):
            print(f"Found normalization module: {name}")

            for key, ft in module.features.items():
                buffer_name = "buffer_" + key.replace(".", "_")
                print(f"  Processing feature: {key}")

                if hasattr(module, buffer_name):
                    buffer = getattr(module, buffer_name)
                    device = next(policy.parameters()).device

                    if 'image' in key:
                        print(f"    Setting image normalization for {key}")
                        # For images: use [0, 1] range (since ToTensor() outputs [0,1])
                        if 'mean' in buffer:
                            buffer['mean'] = torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).to(device)
                        if 'std' in buffer:
                            buffer['std'] = torch.tensor([1.0, 1.0, 1.0]).view(1, 3, 1, 1).to(device)
                        if 'min' in buffer:
                            buffer['min'] = torch.tensor([0.0]).view(1, 3, 1, 1).to(device)
                        if 'max' in buffer:
                            buffer['max'] = torch.tensor([1.0]).view(1, 3, 1, 1).to(device)

                    elif 'state' in key:
                        print(f"    Setting state normalization for {key}")
                        # For robot state: use identity normalization
                        state_dim = ft.shape[-1] if ft.shape else 6  # Default to 6 DOF

                        if 'mean' in buffer:
                            buffer['mean'] = torch.zeros(1, 1, state_dim).to(device)
                        if 'std' in buffer:
                            buffer['std'] = torch.ones(1, 1, state_dim).to(device)
                        if 'min' in buffer:
                            buffer['min'] = torch.full((1, 1, state_dim), -10.0).to(device)
                        if 'max' in buffer:
                            buffer['max'] = torch.full((1, 1, state_dim), 10.0).to(device)

                    else:
                        print(f"    Setting default normalization for {key}")
                        # For other features: use identity normalization
                        if 'mean' in buffer:
                            buffer['mean'] = torch.zeros_like(buffer['mean']).to(device)
                        if 'std' in buffer:
                            buffer['std'] = torch.ones_like(buffer['std']).to(device)
                        if 'min' in buffer:
                            buffer['min'] = torch.zeros_like(buffer['min']).to(device)
                        if 'max' in buffer:
                            buffer['max'] = torch.ones_like(buffer['max']).to(device)

    print("✓ Normalization stats initialized")

if __name__ == "__main__":
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    # print("policy", policy.__dict__.keys())

    print("Initializing normalization statistics...")
    initialize_smolvla_normalization_stats(policy)

    # Set to evaluation mode
    policy.eval()
    # img = Image.open("lerobot/media/lerobot-logo-light.png")
    # If you have a trained policy, you can use it like this:
    # policy = your_trained_policy  # Load your trained policy here

    logo_path = "/home/hartvi/Genesis/lerobot/media/lerobot-logo-light.png"

    batch = create_minimal_batch(image_path=logo_path, image2_path=logo_path, image3_path=logo_path)
    action = inference_with_policy(policy, batch)

    print(f"Predicted action: {action}")

    print("\nTo use with your trained policy:")
    print("1. Load your trained policy model")
    print("2. Call inference_with_policy(policy, batch)")
    print("3. The returned action can be sent to your robot controller")