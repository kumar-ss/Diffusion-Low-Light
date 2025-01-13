import os
import torch
import torch.utils.data
import PIL
from PIL import Image
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor

class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):
        # Get train and test datasets directly from the directory (using 'Test' for validation)
        train_dataset = AllWeatherDataset(
            os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'Train'),
            patch_size=self.config.data.patch_size
        )
        test_dataset = AllWeatherDataset(
            os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'Test'),  # Changed from 'val' to 'test'
            patch_size=self.config.data.patch_size,
            train=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return train_loader, test_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train=True):
        super().__init__()

        self.dir = dir
        self.train = train
        self.patch_size = patch_size

        # Define the directories for 'Low' and 'Normal' images
        self.low_dir = os.path.join(self.dir, 'Low')
        self.normal_dir = os.path.join(self.dir, 'Normal')

        # List all image files from both subdirectories
        self.input_names = [os.path.join('Low', f) for f in os.listdir(self.low_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.input_names += [os.path.join('Normal', f) for f in os.listdir(self.normal_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Corresponding 'ground truth' images in 'Normal' for 'Low' and vice versa
        self.gt_names = [name.replace('Low', 'Normal') for name in self.input_names]

        # Apply transformations (if required)
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        # Load the input image (Low image)
        input_img = Image.open(os.path.join(self.dir, input_name))

        # Check if the ground truth image exists
        gt_path = os.path.join(self.dir, gt_name)
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth image '{gt_name}' not found. Using input image as ground truth.")
            gt_img = input_img  # Use the input image as the ground truth
        else:
            # Load the ground truth image (Normal image)
            gt_img = Image.open(gt_path)

        # Apply transformations to both images
        input_img, gt_img = self.transforms(input_img, gt_img)

        # Concatenate input and ground truth images
        return torch.cat([input_img, gt_img], dim=0), input_name

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)
