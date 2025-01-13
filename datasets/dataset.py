import os
import torch
import torch.utils.data
import PIL
from PIL import Image
import re
from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        # Directly get the paths for 'Low' and 'Normal' images
        train_dataset = AllWeatherDataset(
            os.path.join(self.config.data.data_dir, self.config.data.train_dataset, 'Train'),
            patch_size=self.config.data.patch_size
        )
        val_dataset = AllWeatherDataset(
            os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'Val'),
            patch_size=self.config.data.patch_size,
            train=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.training.batch_size,
            shuffle=True, num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


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
        self.low_images = sorted([f for f in os.listdir(self.low_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.normal_images = sorted([f for f in os.listdir(self.normal_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        # Corresponding 'ground truth' images in 'Normal' for 'Low' and vice versa
        self.input_names = [os.path.join('Low', f) for f in self.low_images]
        self.gt_names = [os.path.join('Normal', f) for f in self.normal_images]

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
        input_name = self.input_names[index].replace('\n', '')
        gt_name = self.gt_names[index].replace('\n', '')
        img_id = re.split('/', input_name)[-1][:-4]

        # Load the images using their paths
        input_img = Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)
