import os
import random
import time
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mushroom_dataset import MushroomDataset
from torch.utils.data import DataLoader


def load_images(base_dir, rgb_only=True):
    images = []
    for genus_folder in os.listdir(base_dir):
        genus_path = os.path.join(base_dir, genus_folder)
        if not os.path.isdir(genus_path):
            continue
        for image in os.listdir(genus_path):
            if not image.endswith((".jpg", ".png", ".jpeg")):
                continue

            image_path = os.path.join(genus_path, image)
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img2:
                    img_copy = img2.copy()
                    images.append({"genus": genus_folder, "image": img_copy})
            except Exception as e:
                print(f"Skipping corrupted image {image_path}: {str(e)}")

    if rgb_only:
        images = [data for data in images if data["image"].mode == "RGB"]
    return images


def encode_labels(images):
    label_encoder = LabelEncoder()
    genera = [data["genus"] for data in images]
    label_encoder.fit(genera)

    for data in images:
        data["encoded_genus"] = label_encoder.transform([data["genus"]])[0]
    return images, label_encoder.classes_


def split_data(data_list, train_size=0.7, val_pct_of_remaining=0.5, random_state=42):
    # Get genera for stratification
    genera = [item["genus"] for item in data_list]

    # First split: separate training data
    train, temp, _, temp_genera = train_test_split(
        data_list,
        genera,
        train_size=train_size,
        stratify=genera,
        random_state=random_state,
    )

    # Second split: divide remaining data into validation and test
    val, test = train_test_split(
        temp,
        test_size=val_pct_of_remaining,
        stratify=temp_genera,
        random_state=random_state,
    )

    return train, val, test


def make_dataloader(data_list, num_augmentations=0, num_workers=1):
    start = time.perf_counter()
    shroom_dataset = MushroomDataset(data_list, num_augmentations=num_augmentations)
    print(f"Dataset creation time: {time.perf_counter()-start}")
    data_loader = DataLoader(
        shroom_dataset, batch_size=32, shuffle=True, num_workers=num_workers
    )
    del data_list
    return data_loader


def create_data_lists(base_dir):
    images = load_images(base_dir=base_dir)
    images, labels = encode_labels(images=images)
    random.shuffle(images)

    train_data, val_data, test_data = split_data(images)
    return train_data, val_data, test_data, labels
