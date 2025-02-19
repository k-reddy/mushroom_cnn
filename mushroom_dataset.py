from torchvision import transforms
from torch.utils.data import Dataset
import random


class MushroomDataset(Dataset):
    def __init__(self, data_list, num_augmentations=0):
        self.data_list = data_list  # Store original data
        self.num_augmentations = num_augmentations
        self.size = 256

        self.base_transform = transforms.Compose(
            [
                transforms.Resize(self.size),
                transforms.CenterCrop((self.size, self.size)),
                transforms.ToTensor(),
            ]
        )

        self.augmentation_list = [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ]

        # Calculate total length without storing data
        self.length = len(data_list) * (1 + num_augmentations)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Calculate which original image to use
        original_idx = idx // (1 + self.num_augmentations)
        is_augmentation = idx % (1 + self.num_augmentations) != 0

        item = self.data_list[original_idx]
        image = self.base_transform(item["image"])

        if is_augmentation:
            num_transforms = random.choice([1, 2])
            # num_transforms = 1
            aug_transform = transforms.Compose(
                random.sample(self.augmentation_list, num_transforms)
            )
            image = aug_transform(image)

        return image, item["encoded_genus"]
