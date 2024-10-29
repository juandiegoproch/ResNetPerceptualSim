import torch
from torchvision import transforms

# Assuming these custom transforms are defined elsewhere
from Transforms import SwapChannels, ElasticTransform, InvertSelectedChannels

# Define the dataset distortions
dataset_distortions = transforms.Compose([
    SwapChannels(mapping=[2, 1, 0]),
    ElasticTransform(alpha_x=5, alpha_y=5, sigma_x=0.5, sigma_y=0.5),
    InvertSelectedChannels(channels_to_invert=[0, 1])
])

# Common transformations applied to all images
common_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),
])

# Unaltered transformations
unaltered_transforms = transforms.Compose([
    common_transform,
])

# Altered transformations
altered_transforms = transforms.Compose([
    dataset_distortions,
    common_transform
])
