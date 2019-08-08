###collections of util function and class for data handling
###list of functions:
###read_from_raw: read raw. file to 1D numpy array
###read_and_reshape: read raw.file and reshape to numpy array
###resample3d: resample a large 3d array by using a sliding window to crop repeatedly, return a list of 3d arrays
###save_as_raw:save a 3d array to raw.file
###resample_and_save3d: resample all raw images in a given directory and save the sub-samples as raw file
###display_from_3d: display a slice from 3d array
###display_from_tensor: display a tensor image (maximum 3 channels)
###slice2d: slice a 3d array along a given axis (x,y,z)


###list of classes:
###LoadTrainDataset_3d: read raw file from given directories and generate pytorch dataset class
###LoadTestDataset_3d: read raw file from given directories and generate pytorch dataset class
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import transforms_3d
from torchvision.transforms import *
from os import listdir
from os.path import join
import random
from math import log10
from scipy import stats
import network_utils
import imageio
import simple_transforms_2d
import os

def read_from_raw(filepath,dtype='uint16'):
    '''
    Read a raw. file and save to numpy array
    '''
    img=np.fromfile(filepath,dtype=dtype)
    return img

def read_and_reshape(filepath,dtype='uint16',dimensions=None):
    '''
    Read a raw. file and reshape into either 2D or 3D numpy array
    
    Inputs:
    
    filepath: filepath of raw. file
    dtype: datatype of raw. file
    dimensions: dimension of the image file. Should be of the format: dx x dy x dz, if none is given, they are inferred from the file name
    
    Output:
    
    img: numpy array representing the image file
    '''
    img=np.fromfile(filepath,dtype=dtype)
    if dimensions==None:
        dimensions = filepath.split('_')[-1].split('.')[0]
    dims=dimensions.split('x')
    if len(dims)==3:
        xdim = int(dims[0])
        ydim = int(dims[1])
        zdim = int(dims[2])
        img=np.reshape(img,(zdim,ydim,xdim))
    else:
        xdim = int(dims[0])
        ydim = int(dims[1])
        img=np.reshape(img,(ydim,xdim))
    return img


     
def resample3d(img,size=(100,100,100),stride=(50,50,50)):
    '''
    re-sample a large 3d image into smaller images by cropping and systematically moving the crop box by a sliding window approach. return a list of 3d arrays.
    '''
    zdim,ydim,xdim=np.shape(img)
    dx,dy,dz=size
    sx,sy,sz=stride
    max_x=xdim-dx
    max_y=ydim-dy
    max_z=zdim-dz
    x=0
    y=0
    z=0
    samples=[]
    while x<=max_x:
        y=0
        while y<=max_y:
            z=0
            while z<=max_z:
                crop_img=img[z:z+dz,y:y+dy,x:x+dx]
                samples.append(crop_img)
                z+=sz
            y+=sy
        x+=sx
    return samples

def resample_and_save3d(image_dir,size=(100,100,100),stride=(50,50,50)):
    image_filenames=[]
    print('reading images from: '+image_dir)
    image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_raw_file(x))
    for i in range(len(image_filenames)):
        img=read_and_reshape(image_filenames[i])
        samples=resample3d(img,size=size,stride=stride)
        for n in range(len(samples)):
            save_as_raw(samples[n],image_filenames[i],dtype='uint16',prefix=str(n))

def crop_and_save3d(image_filename,crop_factor=2):
    img=read_and_reshape(image_filename)
    zdim,ydim,xdim=np.shape(img)
    new_z=zdim//crop_factor
    new_y=ydim//crop_factor
    new_x=xdim//crop_factor
    i=0
    for iz in range(crop_factor):
        for iy in range(crop_factor):
            for ix in range(crop_factor):
                subsample=img[(iz*new_z):((iz+1)*new_z),(iy*new_y):((iy+1)*new_y),(ix*new_x):((ix+1)*new_x)]
                save_as_raw(subsample,image_filename,dtype='uint16',prefix=str(i))
                i+=1
    
def combine_cropped(image_dir,crop_factor=2,dtype='uint8'):
    image_filenames=[]
    print('reading images from: '+image_dir)
    image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_raw_file(x))
    img=read_and_reshape(image_filenames[0],dtype=dtype)
    zdim,ydim,xdim=np.shape(img)
    
    new_z=zdim*crop_factor
    new_y=ydim*crop_factor
    new_x=xdim*crop_factor
    img=np.zeros((new_z,new_y,new_x)).astype(dtype)
    i=0
    for iz in range(crop_factor):
        for iy in range(crop_factor):
            for ix in range(crop_factor):
                print(i)
                subsample=read_and_reshape(image_filenames[i],dtype=dtype)
                img[(iz*zdim):((iz+1)*zdim),(iy*ydim):((iy+1)*ydim),(ix*xdim):((ix+1)*xdim)]=subsample
                i+=1    
    save_as_raw(img,image_filenames[0],dtype=dtype,prefix='combined')

def slice2d(img,axis='z'):
    '''
    Return a list of 2d arrays along a certain axis from a 3d array
    
    Input:
    
    img: 3D numpy array
    axis: axis along which slices will be taken
    
    Output:
    
    slices: list of 2d arrays 
    '''
    zdim,ydim,xdim=np.shape(img)
    slices=[]
    if axis=='z':
        for i in range(zdim):
            slices.append(img[i,:,:])
    elif axis=='y':
        for i in range(ydim):
            slices.append(img[:,i,:])
    else:
        for i in range(xdim):
            slices.append(img[:,:,i])
    return slices

def save_as_bmp(img,filepath,postfix='New'):
    '''
    Take a 2d image and save into bmp image
    
    Inputs:
    
    img: numpy array of the image
    filepath: filepath to save to
    postfix: postfix to the filename which will be attached to the end of filepath
    '''
    img = Image.fromarray(img)
    filepath=filepath+'-'+postfix+'.bmp'
    img.save(filepath)
    
def resample_and_save2d(image_dir,axis='z',dtype='uint16'):
    '''
    Takes 3d images and breaks them down to 2d slices along x,y or z direction and saving the 2d slices
    
    Inputs:
    
    image_dir: directory of the 3d images. All images under this directory should be 3D and will be sliced.
    axis: the axis along which slices will be taken, could be x, y or z
    dtype: file datatype of the raw. file
    
    Output:
    
    x_slice/y_slice/z_slice: the folder where the slices will be saved to
    '''
        
    image_filenames=[]
    print('reading images from: '+image_dir)
    image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_raw_file(x))
    for i in range(len(image_filenames)):
        img=read_and_reshape(image_filenames[i],dtype=dtype)
        samples=slice2d(img,axis=axis)
        for n in range(len(samples)):
            save_as_raw(samples[n],image_filenames[i],dtype='uint16',prefix=str(n),new_dir='%s_slice'%axis) 
    
    
    
    
    
    
def save_as_raw(img,filepath,dtype='uint16',prefix='NEW',new_parent_dir=None,new_dir=None):
    '''
    Takes a 3d or 2d image, flatten it into 1d, and then save to a filepath, the input filepath is the old filepath of the raw image file, this is necessary because we are only potentially changing the dimension of the file and want to preserve other unmodified information about the file. The new file will be in the same directory, with a prefix before the file name. The dimension is also updated in the new file name
    
    Inputs:
    
    img: 3d or 2d image (numpy array)
    filepath: old filepath of the raw image file
    dtype: datatype of the output raw file
    prefix: prefix to be added before the image filename
    new_parent_dir: the new parent directory where image will be saved to. This will be one level higher than the original filepath
    new_dir: new directory where image will be saved to. This will be on the same level as the original filepath
    
    Output:
    
    new raw. file saved to either new_parent_dir or new_dir
    '''
    
    filename=filepath.split('\\')[-1]
    if new_parent_dir != None:
        old_parent_dir=filepath.split('\\')[-2]
        
        old_dir=join(*filepath.split('\\')[:-1])
        newdir=old_dir.replace(old_parent_dir,new_parent_dir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
            
        filepath=filepath.replace(old_dir,newdir)
        
    if new_dir != None:
        dirname=os.path.dirname(filepath)
        newdir=join(dirname,new_dir)
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        dirname=newdir  
        filepath=join(newdir,filename)
        
    old_dimensions=filename.split('_')[-1].split('.raw')[0]
    dims=len(img.shape)
    if dims==3:
        zdim,ydim,xdim=img.shape
    
        zdim=str(zdim)
        ydim=str(ydim)
        xdim=str(xdim)
    
        dimensions=[xdim,ydim,zdim]
        
    else:
        ydim,xdim=img.shape
        ydim=str(ydim)
        xdim=str(xdim)
        dimensions=[xdim,ydim]
        
    new_dimensions='x'.join(dimensions)
    new_filename=filename.replace(old_dimensions,new_dimensions)
    if prefix != None:
        new_filename=prefix+'-'+new_filename
    new_filepath=filepath.replace(filename,new_filename)
    new_img=img.flatten()
    new_img.astype(dtype).tofile(new_filepath)

          
    
def display_from3d(img,slice_direction='z',layer=0):
    '''
    Display a slice of a 3d image given as a 3d array. Slice_direction is the axis that the slice is perpendicular to, layer is the index of the layer of the slice
    
    Input:
    
    img: 3d array
    '''
    if slice_direction=='z':
        img=img[layer,:,:]
    elif slice_direction=='y':
        img=img[:,layer,:]
    else:
        img=img[:,:,layer]
    plt.imshow(img)
     
    
def display_from_tensor(img):
    '''
    Display a 2d image from a given tensor, the tensor need to be in (3,x,y) format or in (1,x,y) format
    
    Input:
    
    img:3d tensor with the first dim 3 or 1
    '''
    npimg=img.numpy()
    if npimg.shape[0]==3:
        plt.imshow(np.transpose(npimg,(1,2,0)))
    else:
        plt.imshow(np.squeeze(npimg))

def is_raw_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)



def PSNR(pred, gt,scale=1.0):
    '''
    Calculate the Peak Signal to Noise Ratio between 2 images
    
    Inputs:
    
    pred: numpy array representing predicted image
    gt: numpy array representing ground truth image
    
    Return:
    
    PSNR value
    '''            
    pred = np.clip(pred,0,scale)
    # pred = (pred - pred.min()) / (pred.max() - pred.min())

    diff = pred - gt
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)

def calculate_average_PSNR(dir1,dir2):
    
    image_filenames_1 = []
    image_filenames_1.extend(join(dir1, x) for x in sorted(listdir(dir1)) if is_raw_file(x))    
    image_filenames_2 = []
    image_filenames_2.extend(join(dir2, x) for x in sorted(listdir(dir2)) if is_raw_file(x))   
    psnr=0
    for i in range(len(image_filenames_1)):
        image_1=read_and_reshape(image_filenames_1[i],dtype='uint8')
        image_2=read_and_reshape(image_filenames_2[i],dtype='uint8')
        psnr+=PSNR(image_1,image_2,scale=1.0)
    psnr=psnr/len(image_filenames_1)
    print('average psnr is:'+str(psnr))
    return psnr
    
def get_residual(image_dir,real_img_fn,pred_img_fn,real_format='png',pred_format='png',segmented=False):
    '''
    Calculate the residual between 2 images
    
    Inputs:
    
    image_dir: directory containing the two images
    real_img_fn: image filename for the first image 
    pred_img_fn: image filename for the second image
    real_format: format for first image (png or raw)
    pred_format: format for second image (png or raw)
    segmented: whether the real_img_fn image is segmented (Boolean)
 
    
    Return:
    
    the residual between the two image saved as png file, a grayscale value close to 128 means the residual is close to 0, a grayscale  value close to 0 or 255 means the residual is large.  
    '''        
    real_img_path=join(image_dir,real_img_fn)
    pred_img_path=join(image_dir,pred_img_fn)
    if real_format=='png':
        real_img=Image.open(real_img_path)
    else:    
        real_img=read_and_reshape(real_img_path)
        if segmented==True:
            real_img=transforms_3d.rescale(real_img,original_scale=(0,1),new_scale=(0,255))
        else:
            real_img=transforms_3d.rescale(real_img,original_scale=(real_img.min(),real_img.max()),new_scale=(0,255))        
    real_img=np.array(real_img).astype(float)    
    real_w,real_h=real_img.shape

    if pred_format=='png':
        pred_img=Image.open(pred_img_path)
    else:    
        pred_img=read_and_reshape(pred_img_path)
        if segmented==True:
            pred_img=transforms_3d.rescale(pred_img,original_scale=(0,1),new_scale=(0,255))
        else:
            pred_img=transforms_3d.rescale(pred_img,original_scale=(pred_img.min(),pred_img.max()),new_scale=(0,255))  
            
    pred_w,pred_h=np.array(pred_img).shape        
    
    if pred_w !=real_w:
        pred_img=pred_img.resize((real_w,real_h),resample=Image.BICUBIC)
    pred_img=np.array(pred_img).astype(float)
    residual=real_img-pred_img
    residual=transforms_3d.rescale(residual,original_scale=(-255,255),new_scale=(0,255)).astype('uint8')
    save_fn =join(image_dir, 'residual_'+pred_img_fn)
    imageio.imwrite(save_fn, residual)                                                                                            
                                                                                              
               


def get_training_set(data_dir, dataset, crop_size, scale_factor,registered=False,grayscale_corrected=True):
    '''
    Return a dataset object for testing
    
    Inputs:
    
    data_dir: directory containing data e.g. 'Rock_registered\\N1.5-N6'
    dataset: a list of dataset to be included in the test dataset such as ['train\\GB1', 'train\\B1M6'] 
    crop_size: size that images will be cropped to  
    scale_factor: scale factor between LR and HR images
    registered: whether HR and LR image pairs are registered (Boolean)
    grayscale_corrected: whether LR image are grayscale histogram matched to HR image (Boolean)
    
    Return:
    
    dataset object
    '''
    if registered:
        hr_train_dir = []
        lr_train_dir = []  
        for data in dataset:
            hr_train_dir.append(join(data_dir, data,'hr'))    
            if grayscale_corrected is True:
                lr_train_dir.append(join(data_dir, data,'Grayscale-adjusted-lr'))
            else:
                lr_train_dir.append(join(data_dir, data,'lr'))
        return LoadRegisteredTrainDataset_2d(hr_train_dir, lr_train_dir,
                                             scale=(0,1.0), 
                                             crop_size=crop_size, 
                                             rotate=True,
                                             fliplr=True, 
                                             fliptb=True, 
                                             scale_factor=scale_factor,
                                            grayscale_corrected=grayscale_corrected)         
        
    else:
        train_dir = []
        for data in dataset:
            train_dir.append(join(data_dir, data,'hr'))
        

        return LoadTrainDataset_2d(train_dir,
                                   scale=(0,1.0),
                                  random_scale=False,    # random scaling
                                  crop_size=crop_size,  # random crop
                                  rotate=True,          # random rotate
                                  fliplr=True,          # random flip
                                  fliptb=True,
                                  scale_factor=scale_factor)

    
    
    
def get_test_set(data_dir, dataset,crop_size,scale_factor,registered=False,grayscale_corrected=True):
    '''
    Return a dataset object for testing
    
    Inputs:
    
    data_dir: directory containing data e.g. Rock_registered\\N1.5-N6
    dataset: a list of dataset to be included in the test dataset such as ['train\GB1', 'train\B1M6'] 
    crop_size: size that images will be cropped to  
    scale_factor: scale factor between LR and HR images
    registered: whether HR and LR image pairs are registered (Boolean)
    grayscale_corrected: whether LR image are grayscale histogram matched to HR image (Boolean)

    Return:
    
    dataset object
    '''
    if registered:
        hr_test_dir = []
        lr_test_dir = []   
        for data in dataset:
            hr_test_dir.append(join(data_dir, data,'hr'))
 
            if grayscale_corrected is True:
                lr_test_dir.append(join(data_dir, data,'Grayscale-adjusted-lr'))
            else:
                lr_test_dir.append(join(data_dir, data,'lr'))
        return LoadRegisteredTestDataset_2d(hr_test_dir, lr_test_dir,
                                             scale=(0,1.0),
                                             crop_size=crop_size, 
                                             scale_factor=scale_factor,
                                           grayscale_corrected=grayscale_corrected)             
    else:
        test_dir = []
        for data in dataset:
            test_dir.append(join(data_dir, data,'hr'))

        return LoadTestDataset_2d(test_dir,
                                   scale=(0,1.0),
                                  crop_size=crop_size,  # random crop
                                  scale_factor=scale_factor)

    
class LoadTrainDataset_2d(data.Dataset):
    '''
    Return a dataset with low-resolution image, high-resolution image, bicubic interpolated image
    Dataloader constructed on the dataset will be a list of low-res images,high-res images and bc images, the low resolution image is bicubic downsampled from high resolution image
    
    Properties:
    
    image_dirs: directory for HR images
    scale: value that all images will be normalized to (tuple)
    randome_scale: whether image will be randomly resized (Boolean)
    crop_size: size that images will be cropped to
    rotate: whether images will be randomly rotated 90, 180 or 270 degrees (Boolean)
    fliplr: whether images will be randomly flipped left to right (Boolean)
    fliptb: whether images will be randomly flipped top to bottom (Boolean)
    scale_factor: scale factor between LR and HR images      
    '''    
    def __init__(self, image_dirs,scale=(0,1.0),random_scale=False, crop_size=100, rotate=True,fliplr=True, fliptb=True, scale_factor=4):
        super(LoadTrainDataset_2d, self).__init__()

        self.image_filenames = []
        for image_dir in image_dirs:
            print('reading images from: '+image_dir)
            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_raw_file(x))
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.rotate=rotate
        self.scale_factor = scale_factor
        self.scale=scale
    def __getitem__(self, index):  
        # load image
        img = read_and_reshape(self.image_filenames[index]).astype(float)    
        img = transforms_3d.rescale(img,original_scale=(0,65536),new_scale=self.scale)
        img = Image.fromarray(img)
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
    
    
class LoadTestDataset_2d(data.Dataset):
    '''
    Return a dataset with low-resolution image, high-resolution image, bicubic interpolated image
    Dataloader constructed on the dataset will be a list of low-res images,high-res images and bc images, the low resolution image is bicubic downsampled from high resolution image
    
    Properties:
    
    image_dirs: directory for HR images
    scale: value that all images will be normalized to (tuple)
    crop_size: size that images will be cropped to
    scale_factor: scale factor between LR and HR images         
    '''    
    def __init__(self, image_dirs,scale=(0,1.0),crop_size=100,scale_factor=4):
        super(LoadTestDataset_2d, self).__init__()

        self.image_filenames = []
        for image_dir in image_dirs:
            print('reading images from: '+image_dir)
            self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_raw_file(x))
        self.scale_factor = scale_factor
        self.scale=scale
        self.crop_size=crop_size
    def __getitem__(self, index):
        # load image
        img = read_and_reshape(self.image_filenames[index]).astype(float)            
        img = transforms_3d.rescale(img,original_scale=(0,65536),new_scale=self.scale)
        # original HR image size
        if self.crop_size is not None:
            self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        else:
            self.crop_size=img.shape[1]
        
        img = Image.fromarray(img)

        hr_img_w = self.crop_size
        hr_img_h = self.crop_size
        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor
        
        transform = RandomCrop(self.crop_size)
        img = transform(img)
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
     

        
def get_registered_slice_index(index_lr,ref_lr,ref_hr,res_lr,res_hr,vsize_lr,vsize_hr):
    '''
    Get registered slice index in a high resolution image from the index of the low resolution image
    
    Input:
    
    index_lr: index of LR image slice
    ref_lr, ref_hr: registered index number for LR and HR (python index)
    res_lr,res_hr: total slice number for LR and HR image
    vsize_lr,vsize_hr: voxel size for LR and HR image
    
    Output:
    
    index_hr: corresponding index on HR image
    '''
    dist1=index_lr-ref_lr
    dist2=dist1*vsize_lr/vsize_hr
    index_hr=int(round(ref_hr+dist2))
    
    if index_hr<0 or index_hr>=res_hr:
        return None
    else:
        return index_hr
    
    
def get_registered_pair(filepath_lr,filepath_hr,ref_lr,ref_hr,dtype='uint16',parent_dir=None,resize=None,scale_factor=2):
    '''
    Take LR and HR image and convert to corresponding 2D slice pairs
    
    Inputs:
    
    filepath_lr: filepath for LR image
    filepath_hr: filepath for HR image
    ref_lr,ref_hr: reference registered LR and HR slice index
    dtype: filetype for LR and HR image
    parent_dir: name of the parent directory that the slices will be saved to
    resize: tuple of the original N value of LR image and the N value to be adjusted to by bicubic interpolation
    scale_factor: desired scale factor between LR and HR image
    
    Outputs:
    parent_dir/lr: folder where LR image slices are saved
    parent_dir/hr: folder where HR image slices are saved
    '''    
    dimensions_lr = filepath_lr.split('_')[-1].split('.raw')[0]
    res_lr=int(dimensions_lr.split('x')[0])
    vsize_lr=float(filepath_lr.split('_')[-4].split('um')[0])
    dimensions_hr = filepath_hr.split('_')[-1].split('.raw')[0]
    res_hr=int(dimensions_hr.split('x')[0])
    vsize_hr=float(filepath_hr.split('_')[-4].split('um')[0])

    img_lr=read_and_reshape(filepath_lr,dtype=dtype)
    img_hr=read_and_reshape(filepath_hr,dtype=dtype)
    
    for i in range(res_lr):
        index_hr=get_registered_slice_index(i,ref_lr,ref_hr,res_lr,res_hr,vsize_lr,vsize_hr)
        if index_hr==None:
            continue
        
        lr_size=int(res_lr*resize[1]/resize[0])
        hr_size=lr_size*scale_factor
        image_lr=simple_transforms_2d.resize2d(img_lr[i,:,:],(lr_size,lr_size))
        image_hr=simple_transforms_2d.resize2d(img_hr[index_hr,:,:],(hr_size,hr_size))
        save_as_raw(image_lr,filepath_lr,prefix='lr-'+str(i)+'-',new_dir=join(parent_dir,'lr'))
        save_as_raw(image_hr,filepath_hr,prefix='hr-'+str(i)+'-',new_dir=join(parent_dir,'hr'))        
            
    
class LoadRegisteredTrainDataset_2d(data.Dataset):
    '''
    Return a dataset with low-resolution image, high-resolution image, bicubic interpolated image from LR image
    Dataloader constructed on the dataset will be a list of low-res images,high-res images and bc images, the low resolution and high    resolution dataset are registered
    
    Properties:
    
    hr_image_dirs: directory for HR images
    lr_image_dirs: directory for LR images
    scale: value that all images will be normalized to (tuple)
    crop_size: size that images will be cropped to
    rotate: whether images will be randomly rotated 90, 180 or 270 degrees (Boolean)
    fliplr: whether images will be randomly flipped left to right (Boolean)
    fliptb: whether images will be randomly flipped top to bottom (Boolean)
    scale_factor: scale factor between LR and HR images
    grayscale_corrected: whether grayscale histogram matched LR image will be used (Boolean)    
    '''
    def __init__(self, hr_image_dirs, lr_image_dirs,scale=(0,1.0), crop_size=100, rotate=True,fliplr=True, fliptb=True, scale_factor=4,grayscale_corrected=False):
        super(LoadRegisteredTrainDataset_2d, self).__init__()

        self.hr_image_filenames = []
        self.lr_image_filenames = []
        for hr_image_dir in hr_image_dirs:
            print('reading high resolution images from: '+hr_image_dir)
            self.hr_image_filenames.extend(join(hr_image_dir, x) for x in sorted(listdir(hr_image_dir)) if is_raw_file(x))
        for lr_image_dir in lr_image_dirs:
            print('reading low resolution images from: '+lr_image_dir)
            self.lr_image_filenames.extend(join(lr_image_dir, x) for x in sorted(listdir(lr_image_dir)) if is_raw_file(x))
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.rotate=rotate
        self.scale_factor = scale_factor
        self.scale=scale
        self.grayscale_corrected=grayscale_corrected
    def __getitem__(self, index):
        # load image
        hr_img = read_and_reshape(self.hr_image_filenames[index]).astype(float)
        lr_img = read_and_reshape(self.lr_image_filenames[index]).astype(float)
                    
        hr_img = transforms_3d.rescale(hr_img,original_scale=(hr_img.min(),hr_img.max()),new_scale=self.scale)
        lr_img = transforms_3d.rescale(lr_img,original_scale=(lr_img.min(),lr_img.max()),new_scale=self.scale)
        hr_img_w,hr_img_h = hr_img.shape
        hr_img = np.clip(hr_img,0,1.0)
        hr_img = Image.fromarray(hr_img)
        
        lr_img_w,lr_img_h = lr_img.shape
        lr_img = np.clip(lr_img,0,1.0)
        lr_img = Image.fromarray(lr_img)
        
        
        # determine LR image size
        lr_img_w_temp = hr_img_w // self.scale_factor
        lr_img_h_temp = hr_img_h // self.scale_factor
        
        if lr_img_w_temp<lr_img_w: #downsample low resolution image

            transform = Resize((lr_img_w_temp, lr_img_h_temp), interpolation=Image.BICUBIC)
            lr_img = transform(lr_img)

            transform= Resize((lr_img_w_temp*self.scale_factor, lr_img_h_temp*self.scale_factor), interpolation=Image.BICUBIC)
            hr_img = transform(hr_img)

        else:#downsample high resolution image:

            transform= Resize((lr_img_w*self.scale_factor, lr_img_h*self.scale_factor), interpolation=Image.BICUBIC)
            hr_img = transform(hr_img)

        #at this stage,the size ratio between hr and lr iamge is exactly scale_factor
        hr_img_w,hr_img_h = hr_img.size
        lr_img_w,lr_img_h = lr_img.size
        
        

        
        x_start_lr=random.randint(0,lr_img_w-self.crop_size/self.scale_factor)
        y_start_lr=random.randint(0,lr_img_h-self.crop_size/self.scale_factor)
        
        lr_img=lr_img.crop((x_start_lr,y_start_lr,x_start_lr+self.crop_size/self.scale_factor,y_start_lr+self.crop_size/self.scale_factor))
        hr_img=hr_img.crop((x_start_lr*self.scale_factor,y_start_lr*self.scale_factor,x_start_lr*self.scale_factor+self.crop_size,
                                      y_start_lr*self.scale_factor+self.crop_size))
        
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            lr_img = lr_img.rotate(90 * rv, expand=True)
            hr_img = hr_img.rotate(90 * rv, expand=True)
        # random horizontal flip
        if self.fliplr:
            if random.random() < 0.5:
                lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)

        # hr_img HR image
        totensor_transform = Compose([ToTensor()])
        hr_img = totensor_transform(hr_img)

        # lr_img LR image
        lr_img =  totensor_transform(lr_img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img
        
    def __len__(self):
        return len(self.hr_image_filenames)    


class LoadRegisteredTestDataset_2d(data.Dataset):
    '''
    Return a dataset with low-resolution image, high-resolution image, bicubic interpolated image from LR image
    Dataloader constructed on the dataset will be a list of low-res images,high-res images and bc images, the low resolution and high    resolution dataset are registered
    
    Properties:
    
    hr_image_dirs: directory for HR images
    lr_image_dirs: directory for LR images
    scale: value that all images will be normalized to (tuple)
    crop_size: size that images will be cropped to
    scale_factor: scale factor between LR and HR images
    grayscale_corrected: whether grayscale histogram matched LR image will be used (Boolean)
    '''
    def __init__(self, hr_image_dirs, lr_image_dirs,scale=(0,1.0),crop_size=100, scale_factor=4,grayscale_corrected=False):
        super(LoadRegisteredTestDataset_2d, self).__init__()

        self.hr_image_filenames = []
        self.lr_image_filenames = []
        for hr_image_dir in hr_image_dirs:
            print('reading high resolution images from: '+hr_image_dir)
            self.hr_image_filenames.extend(join(hr_image_dir, x) for x in sorted(listdir(hr_image_dir)) if is_raw_file(x))
        for lr_image_dir in lr_image_dirs:
            print('reading low resolution images from: '+lr_image_dir)
            self.lr_image_filenames.extend(join(lr_image_dir, x) for x in sorted(listdir(lr_image_dir)) if is_raw_file(x))
 
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.scale=scale
        self.grayscale_corrected=grayscale_corrected
    def __getitem__(self, index):
        hr_img = read_and_reshape(self.hr_image_filenames[index]).astype(float)
        lr_img = read_and_reshape(self.lr_image_filenames[index]).astype(float)
             
        hr_img = transforms_3d.rescale(hr_img,original_scale=(hr_img.min(),hr_img.max()),new_scale=self.scale)
        lr_img = transforms_3d.rescale(lr_img,original_scale=(lr_img.min(),lr_img.max()),new_scale=self.scale)                   
        hr_img_w,hr_img_h = hr_img.shape
        hr_img = np.clip(hr_img,0,1.0)
        hr_img = Image.fromarray(hr_img)
        
        lr_img_w,lr_img_h = lr_img.shape
        lr_img = np.clip(lr_img,0,1.0)
        lr_img = Image.fromarray(lr_img)
        
        if self.crop_size is None:
            self.crop_size = hr_img_w
        
        # determine LR image size
        lr_img_w_temp = hr_img_w // self.scale_factor
        lr_img_h_temp = hr_img_h // self.scale_factor
        
        
        if lr_img_w_temp<lr_img_w: #downsample low resolution image

            transform = Resize((lr_img_w_temp, lr_img_h_temp), interpolation=Image.BICUBIC)
            lr_img = transform(lr_img)

            transform= Resize((lr_img_w_temp*self.scale_factor, lr_img_h_temp*self.scale_factor), interpolation=Image.BICUBIC)
            hr_img = transform(hr_img)

        else:#downsample high resolution image:

            transform= Resize((lr_img_w*self.scale_factor, lr_img_h*self.scale_factor), interpolation=Image.BICUBIC)
            hr_img = transform(hr_img)


        #at this stage,the size ratio between hr and lr iamge is exactly scale_factor
        hr_img_w,hr_img_h = hr_img.size
        lr_img_w,lr_img_h = lr_img.size
        
        x_start_lr=random.randint(0,lr_img_w-self.crop_size/self.scale_factor)
        y_start_lr=random.randint(0,lr_img_h-self.crop_size/self.scale_factor)
        
        lr_img=lr_img.crop((x_start_lr,y_start_lr,x_start_lr+self.crop_size/self.scale_factor,y_start_lr+self.crop_size/self.scale_factor))
        hr_img=hr_img.crop((x_start_lr*self.scale_factor,y_start_lr*self.scale_factor,x_start_lr*self.scale_factor+self.crop_size,
                                      y_start_lr*self.scale_factor+self.crop_size))
        
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

  

        # hr_img HR image
        totensor_transform = Compose([ToTensor()])
        hr_img = totensor_transform(hr_img)

        # lr_img LR image
        lr_img =  totensor_transform(lr_img)

        # Bicubic interpolated image
        bc_transform = Compose([ToPILImage(), Resize((hr_img_w, hr_img_h), interpolation=Image.BICUBIC), ToTensor()])
        bc_img = bc_transform(lr_img)

        return lr_img, hr_img, bc_img
        
    def __len__(self):
        return len(self.hr_image_filenames)    


     

        
def adjust_grayscale(hr_dir,lr_dir):
    '''
    Adjust images by shifting the histogram to match the other image in image pairs 
    
    Input: 
    
    lr_dir: directory containing images to be adjusted(2D raw file)
    hr_dir: directory containing images for adjustment reference (2D raw file)
    
    Output:
    
    Grayscale-adjusted-lr: folder contaning grayscale adjusted raw images 
    
    '''        
    hr_image_filenames = []
    lr_image_filenames = []

    hr_image_filenames.extend(join(hr_dir, x) for x in sorted(listdir(hr_dir)) if is_raw_file(x))

    lr_image_filenames.extend(join(lr_dir, x) for x in sorted(listdir(lr_dir)) if is_raw_file(x))
    

    
    for index in range(len(hr_image_filenames)):
                # load image
        hr_img = read_and_reshape(hr_image_filenames[index]).astype(float)
        hr_img_w,hr_img_h = hr_img.shape
        percentile=np.linspace(0,100,5000)
        hr_img_old=hr_img
        
        lr_img = read_and_reshape(lr_image_filenames[index]).astype(float)
        lr_img_w,lr_img_h = lr_img.shape
        lr_img_old=lr_img

        
        hr_percentile=np.percentile(hr_img_old,percentile)
        lr_percentile=np.percentile(lr_img,percentile)
        
        
        for i in range(lr_img_w):
            for j in range(lr_img_h):
                lr_img[i,j]=np.interp(lr_img_old[i,j],lr_percentile,hr_percentile)
        
    
        save_as_raw(lr_img,lr_image_filenames[index],dtype='uint16',prefix='Grayscale-adjusted',new_parent_dir='Grayscale-adjusted-lr')
    
def image_expansion_crop(image_dir,expand_size=1,dtype='uint16'):
    '''
    Adjust images by expanding the image using nearest neighbor method and then cropping to original size 
    
    Input: 
    
    image_dir: directory containing images (2D raw file)
    expand_size: size for the expansion and cropping (postive if HR images are adjusted, negative if LR images are adjusted) 
    
    Output:
    
    Expansion-adjusted-hr/Expansion-adjusted-lr: folder contaning expansion adjusted 2D HR/LR raw images 
    
    '''        
    image_filenames = []

    image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_raw_file(x))
    
    for image_fn in image_filenames:
        img=read_and_reshape(image_fn,dtype=dtype).astype(float)
        img_w,img_h=img.shape
        if expand_size>=0:
            img=simple_transforms_2d.resize2d(img,(img_w+2*expand_size,img_h+2*expand_size),resample='NEAREST')
            img=img[expand_size:expand_size+img_w,expand_size:expand_size+img_h]
            save_as_raw(img,image_fn,dtype='uint16',prefix=None)
        else:
            expand_size=-expand_size
            img=simple_transforms_2d.resize2d(img,(img_w+2*expand_size,img_h+2*expand_size),resample='NEAREST')
            img=img[expand_size:expand_size+img_w,expand_size:expand_size+img_h]
            save_as_raw(img,image_fn,dtype='uint16',prefix=None)            
    
    
def calculate_expansion_size(hr_dir,lr_dir,size_list=[-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9],dtype='uint16'):
    '''
    Calculate the best adjustment size so that HR and LR images are aligned at the corners. This calculation is done between grayscale histogram matched LR and HR image pairs. 5 LR and HR image pairs are randomly selected. The images are first expanded and then cropped back to original size. Essentially this corresponds to cropping the outer rim of the image without changing the size of the image. This is done because most image pairs are aligned at the center but starts to misalign at the corner. A series of values for the cropped outer rim width are tested (positive value for cropping HR image and negative value for cropping LR image). For each value tested, the residual between LR and resized&cropped HR or HR and resized&cropped LR is calculated. The value with lowest standard deviation is selected as the best size for the adjustment.
    
    Input: 
    
    lr_dir: directory containing registered LR images (2D raw file)
    hr_dir: directory containing registered HR images (2D raw file)
    size_list: a list of values to test (must be integer)
    
    Output:
 
    best size for cropping and expansion adjustment (positive value: adjusting HR images, negative value: adjusting LR images)
    '''    
    hr_image_filenames = []
    lr_image_filenames = []

    hr_image_filenames.extend(join(hr_dir, x) for x in sorted(listdir(hr_dir)) if is_raw_file(x))
    
    lr_image_filenames.extend(join(lr_dir, x) for x in sorted(listdir(lr_dir)) if is_raw_file(x))
    
    average_stdlist=np.zeros(len(size_list))
    indices=np.random.randint(0,len(hr_image_filenames),size=5)
    for i,size in enumerate(size_list):
        stdlist=[]
        if size>=0:
            for index in indices:
                hr_img=read_and_reshape(hr_image_filenames[index],dtype=dtype).astype(float)
                lr_img=read_and_reshape(lr_image_filenames[index],dtype=dtype).astype(float)
                hr_img_w,hr_img_h = hr_img.shape
                lr_img_w,lr_img_h = lr_img.shape
                lr_img_reshaped=simple_transforms_2d.resize2d(lr_img,(hr_img_w,hr_img_h),resample='NEAREST')
                hr_img_reshaped=simple_transforms_2d.resize2d(hr_img,(hr_img_w+2*size,hr_img_h+2*size),resample='NEAREST')
                hr_img_reshaped=hr_img_reshaped[size:size+hr_img_w,size:size+hr_img_h]
                residual=hr_img_reshaped-lr_img_reshaped
                stdlist.append(np.std(residual))
            average_stdlist[i]=np.mean(stdlist)
            print('adjust HR images size:'+str(size))
            print('std:'+str(average_stdlist[i]))
        else:
            for index in indices:                
                hr_img=read_and_reshape(hr_image_filenames[index],dtype=dtype).astype(float)
                lr_img=read_and_reshape(lr_image_filenames[index],dtype=dtype).astype(float)
                hr_img_w,hr_img_h = hr_img.shape
                lr_img_w,lr_img_h = lr_img.shape
                
                
              
                lr_img_reshaped=simple_transforms_2d.resize2d(lr_img,(lr_img_w+2*abs(size),lr_img_h+2*abs(size)),resample='NEAREST')
                lr_img_reshaped=lr_img_reshaped[abs(size):abs(size)+lr_img_w,abs(size):abs(size)+lr_img_h]
                lr_img_reshaped=simple_transforms_2d.resize2d(lr_img_reshaped,(hr_img_w,hr_img_h),resample='NEAREST')
                residual=hr_img-lr_img_reshaped
                stdlist.append(np.std(residual))
            average_stdlist[i]=np.mean(stdlist)
            print('adjust LR images size:'+str(-size))
            print('std:'+str(average_stdlist[i]))
    return size_list[np.argmin(average_stdlist)]
    

    
def image_correction(hr_dir,lr_dir,dtype='uint16'):
    '''
    Perform grayscale histogram matching and expansion adjustment. Grayscale matching will adjust the LR image so that the histogram match the HR image. Expansion adjustment will adjust either the HR or LR so that they are aligned at the center.
    
    Input: 
    
    lr_dir: directory containing registered LR images (2D raw file)
    hr_dir: directory containing registered HR images (2D raw file)

    Output:
 
    Grayscale-adjusted-lr: folder containing grayscale histogram matched 2D LR raw images
    Expansion-adjusted-hr/Expansion-adjusted-lr: folder contaning expansion adjusted 2D HR/LR raw images
    '''    
    adjust_grayscale(hr_dir,lr_dir)
    grayscale_adjusted_lr_dir=lr_dir.replace(lr_dir.split('\\')[-1],'Grayscale-adjusted-lr')
    size=calculate_expansion_size(hr_dir,grayscale_adjusted_lr_dir,dtype=dtype)
    if size>=0:
        print('the adjusted size for HR images is:'+str(size))
        image_expansion_crop(hr_dir,expand_size=size,dtype=dtype)
    else:
        print('the adjusted size for LR images is:'+str(-size))
        image_expansion_crop(grayscale_adjusted_lr_dir,expand_size=-size,dtype=dtype)



        
def register_and_correct(filepath_lr,filepath_hr,N_original,N_adjust,ref,dtype='uint16'):
    '''
    Register a LR and HR image pair and perform grayscale histogram matching and expansion adjustment. Grayscale matching will adjust the LR image so that the histogram match the HR image. Expansion adjustment will adjust either the HR or LR so that they are aligned at the center. Both images will also be resized to new pair of N values (usually an integer value). This is done to bin different N values to the same number. 
    
    Input: 
    
    filepath_lr: file location for LR image (3D raw file)
    filepath_hr: file location for HR image (3D raw file)
    N_original: a tuple of original N value for LR and HR image.
    N_adjust: a tuple of the desired N value for LR and HR image, this is the N value pair that the images will be resized to using bicubic               interpolation
    ref: a tuple of the index of the registered slice for LR and HR image. This is the python index and starts from 0.
    
    Output:
    
    lr: folder containing 2D LR raw images (slices from original 3D raw file)
    hr: folder containing 2D HR raw images (slices from original 3D raw file)
    Grayscale-adjusted-lr: folder containing grayscale histogram matched 2D LR raw images
    Expansion-adjusted-hr/Expansion-adjusted-lr: folder contaning expansion adjusted 2D HR/LR raw images
    '''
    
    
    N_lr,N_hr=N_adjust
    N_lr_original,N_hr_original=N_original
    scale_factor=int(N_hr/N_lr)
    ref_lr,ref_hr=ref
    parent_dir=os.path.dirname(filepath_lr)
    resultdir=os.path.join(parent_dir,'%s-%s'%(N_lr,N_hr))
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)
    get_registered_pair(filepath_lr,filepath_hr,ref_lr,ref_hr,dtype=dtype,parent_dir='%s-%s'%(N_lr,N_hr),resize=(N_lr_original,N_lr),scale_factor=scale_factor)
    hr_dir=os.path.join(resultdir,'hr')
    lr_dir=os.path.join(resultdir,'lr')
    image_correction(hr_dir,lr_dir,dtype=dtype)
    
  
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    