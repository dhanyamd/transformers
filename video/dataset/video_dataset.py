import glob
import os
import cv2
import torch
import numpy as np
import random
import torchvision
import pickle
import torchvision.transforms.v2
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class VideoDataset(Dataset):
    r"""
    Simple Video Dataset class for training diffusion transformer
    for video generation.
    If latents are present, the dataset uses the saved latents for the videos,
    else it reads the video and extracts frames from it.
    """
    def __init__(self, split, dataset_config, latent_path=None, im_ext='png'):
        r"""
        Initialize all parameters and also check
        if latents are present or not
        :param split: for now this is always train
        :param dataset_config: config parameters for dataset(mnist/ucf)
        :param latent_path: Path for saved latents
        :param im_ext: assumes all images are of this extension. Used only
        if latents are not present
        """
        self.split = split
        self.video_ext = dataset_config['video_ext']
        self.num_images = dataset_config['num_images_train']
        self.use_images = self.num_images > 0
        self.num_frames = dataset_config['num_frames']
        self.frame_interval = dataset_config['frame_interval']
        self.frame_height = dataset_config['frame_height']
        self.frame_width = dataset_config['frame_width']
        self.frame_channels = dataset_config['frame_channels']
        self.center_square_crop = dataset_config['centre_square_crop']
        self.filter_fpath = dataset_config['video_filter_path']
        if self.center_square_crop:
            assert self.frame_height == self.frame_width, \
                ('For centre square crop frame_height '
                 'and frame_width should be same')
            self.transforms = torchvision.transforms.v2.Compose([
                    torchvision.transforms.v2.Resize(self.frame_height),
                    torchvision.transforms.v2.CenterCrop(self.frame_height),
                    torchvision.transforms.v2.ToPureTensor(),
                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                    torchvision.transforms.v2.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transforms = torchvision.transforms.v2.Compose([
                    torchvision.transforms.v2.Resize((self.frame_height,
                                                      self.frame_width)),
                    torchvision.transforms.v2.ToPureTensor(),
                    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                    torchvision.transforms.v2.Normalize(mean=[0.5, 0.5, 0.5],
                                                        std=[0.5, 0.5, 0.5])
                ])
