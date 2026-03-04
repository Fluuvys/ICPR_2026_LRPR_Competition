"""Augmentation pipelines for training, validation, and degradation."""
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the sequence targets once to reuse across all pipelines
SEQUENCE_TARGETS = {
    'image1': 'image', 
    'image2': 'image', 
    'image3': 'image', 
    'image4': 'image'
}

def get_train_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    """Photometric and Spatial only. No heavy blur to avoid double-corruption."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Affine(scale=(0.98, 1.02), translate_percent=(0.02, 0.02), rotate=(-3, 3), fill=128, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=0.3),
        
        # Dialed way down to protect thin characters like "1" or "7"
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(2, 4), hole_width_range=(2, 4), fill=128, p=0.1),
        
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ], additional_targets=SEQUENCE_TARGETS)

def get_light_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ], additional_targets=SEQUENCE_TARGETS)

def get_degradation_transforms() -> A.Compose:
    """Optical destruction. Applied at HIGH RESOLUTION before downscaling."""
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0)
        ], p=0.7),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0)
        ], p=0.7),
        A.ImageCompression(quality_range=(20, 60), p=0.5),
        # Notice: No A.Resize here. We degrade at native resolution.
    ], additional_targets=SEQUENCE_TARGETS)

def get_val_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ], additional_targets=SEQUENCE_TARGETS)