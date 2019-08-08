import torch.utils.data as data
from torchvision.transforms import *
from os import listdir
from os.path import join
from PIL import Image
import random
from torch.utils.data import DataLoader
import utils
import matplotlib.pyplot as plt
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDatasetFromFolder(data.Dataset):
    '''
    return a dataset with low-resolution image, high-resolution image, bicubic interpolated image
    dataloader constructed on the dataset will be a list of low-res images,high-res images and bc images
    '''
    def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=128, rotate=True, fliplr=True, fliptb=True, scale_factor=4):
        super(TrainDatasetFromFolder, self).__init__()

        self.image_filenames = []
        for image_dir in image_dirs:
            print('reading images from: '+image_dir)
            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random resize between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)
            transform = Resize((scale_w, scale_h), interpolation=Image.BICUBIC)
            img = transform(img)

        # random crop
        transform = RandomCrop(self.crop_size)
        img = transform(img)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            img, _, _ = img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, is_gray=False,scale_factor=4):
        super(TestDatasetFromFolder, self).__init__()
        self.image_filenames = []
        for image_dir in image_dirs:
            print('reading images from: '+image_dir)
            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        self.scale_factor = scale_factor
        self.is_gray=is_gray
    def __getitem__(self, index):
        # load image
        img = load_img(self.image_filenames[index])

        # original HR image size
        w = img.size[0]
        h = img.size[1]

        # determine valid HR image size with scale factor
        hr_img_w = calculate_valid_crop_size(w, self.scale_factor)
        hr_img_h = calculate_valid_crop_size(h, self.scale_factor)

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img = img.convert('YCbCr')
            img, _, _ = lr_img.split()

        # hr_img HR image
        hr_transform = Compose([Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        hr_img = hr_transform(img)

        # lr_img LR image
        lr_transform = Compose([Resize((lr_img_w, lr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        lr_img = lr_transform(img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.image_filenames)

    
def get_training_set(data_dir, dataset, crop_size, scale_factor, is_gray=False):
    train_dir = []
    if dataset == 'bsds500':
        train_dir.append('test_data\\BSR\\BSDS500\\data\\images\\train')
    else:
        train_dir.append(join(data_dir, dataset))

    return TrainDatasetFromFolder(train_dir,
                                  is_gray=is_gray,
                                  random_scale=True,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)


def get_test_set(data_dir, dataset, scale_factor, is_gray=False):
    test_dir = []
    if dataset == 'bsds500':
        test_dir.append('test_data\\BSR\\BSDS500\\data\\images\\test')
    else:
        test_dir.append(join(data_dir, dataset))

    return TestDatasetFromFolder(test_dir,
                                 is_gray=is_gray,
                                 scale_factor=scale_factor)    
    
    
    
    
def get_dataloader(dataset,batch_size=32,shuffle=True):
    '''
    generate a dataloader from a given dataset
    '''
    dataloader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
    return dataloader

def plot_from_dataloader(dataloader,plot_size=2,figsize=(64,64)):
    '''
    sample and dataloader and plot a few examples from it. Each row has 3 columns: low resolution image, high resolution image,bicubic interpolated image from low resolution images
    '''
    dataiter=iter(dataloader)
    images=dataiter.next()
    #images is a list of lr, hr and bc images, each element has the shape of (batch_size,3,xsize,ysize)
    lr_images=images[0]
    hr_images=images[1]
    bc_images=images[2]
    
    fig=plt.figure(figsize=figsize)
    for idx in np.arange(plot_size):
        ax=fig.add_subplot(plot_size,3,3*idx+1,xticks=[],yticks=[])
        utils.display_from_tensor(lr_images[idx])
        ax=fig.add_subplot(plot_size,3,3*idx+2,xticks=[],yticks=[])
        utils.display_from_tensor(hr_images[idx])
        ax=fig.add_subplot(plot_size,3,3*idx+3,xticks=[],yticks=[])
        utils.display_from_tensor(bc_images[idx])



