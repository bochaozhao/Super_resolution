###crop3d: crop a 3d array and return a 3d array
###rescale: rescale a 3d array


import numpy as np
import scipy
import random
import utils
import os
import psutil



def crop3d(img,range_x=None,range_y=None,range_z=None):
    '''
    Crop a 3d numpy array according to desired range along each axis, if None is input, no cropping is done on that dimension,each input is a tuple of start and end index
    
    Inputs:
    
    img: 3D numpy array
    range_x, range_y, range_z: tuple of cropping range along x, y and z axis
    
    Return:
    
    new 3D numpy array
    '''
    zdim,ydim,xdim=np.shape(img)
    if range_z is not None:
        img=img[range_z[0]:range_z[1]+1,:,:]
    if range_y is not None:
        img=img[:,range_y[0]:range_y[1]+1,:]
    if range_x is not None:
        img=img[:,:,range_x[0]:range_x[1]+1]
    return img


def RandomCrop3d(img,crop_size):
    zdim,ydim,xdim=np.shape(img)
    zmax=zdim-crop_size
    ymax=ydim-crop_size
    xmax=xdim-crop_size
    xstart=random.randint(0, xmax)
    ystart=random.randint(0, ymax)
    zstart=random.randint(0, zmax)
    img=img[zstart:zstart+crop_size,ystart:ystart+crop_size,xstart:xstart+crop_size]
    return img

def RandomHorizontalFlip(img):
    if random.random() < 0.5:
        img=np.flip(img,axis=1)
    if random.random() < 0.5:
        img=np.flip(img,axis=2)
    return img

def RandomVerticalFlip(img):
    if random.random() < 0.5:
        img=np.flip(img,axis=0)
    return img


def rescale(img,original_scale=(0,65536),new_scale=(0,1.0)):
    '''
    Rescale an image from original_scale to new_scale
    '''
    img=img.astype(float)
    old_min,old_max=original_scale
    new_min,new_max=new_scale
    img_new=new_min+(new_max-new_min)/(old_max-old_min)*(img-old_min)
    return img_new


def combine_2d_slices(dirs,scale_factor=4):
    '''
    Combine 2D slices to form 3D volumes
    '''
    ###z slice
    img_dir=os.path.join(dirs,'z_slice')
    images=os.listdir(img_dir)
    num_images=len(images)
    dim=int(images[0].split('_')[-1].split('x')[0])
    
    interpolated=np.zeros((dim,dim,dim)).astype('uint16')
    combined=np.zeros((num_images,dim,dim)).astype(float)


    for i in range(num_images):
        image=utils.read_and_reshape(os.path.join(img_dir,images[i]))
        index=int(images[i].split('-')[1])
        combined[index,:,:]=image
        print(i)
    
    if scale_factor==4:
        for i in range(num_images):
            if i==0:
                interpolated[4*i,:,:]+=((combined[i,:,:]-3*(combined[i+1,:,:]-combined[i,:,:])/8)/3).astype('uint16')
                interpolated[4*i+1,:,:]+=((combined[i,:,:]-(combined[i+1,:,:]-combined[i,:,:])/8)/3).astype('uint16')       
            else:    
                interpolated[4*i,:,:]+=((combined[i,:,:]-3*(combined[i,:,:]-combined[i-1,:,:])/8)/3).astype('uint16')
                interpolated[4*i+1,:,:]+=((combined[i,:,:]-(combined[i,:,:]-combined[i-1,:,:])/8)/3).astype('uint16')
    
            if i==(num_images-1):
                interpolated[4*i+2,:,:]+=((combined[i,:,:]+(combined[i,:,:]-combined[i-1,:,:])/8)/3).astype('uint16')
                interpolated[4*i+3,:,:]+=((combined[i,:,:]+3*(combined[i,:,:]-combined[i-1,:,:])/8)/3).astype('uint16')
            else:    
                interpolated[4*i+2,:,:]+=((combined[i,:,:]+(combined[i+1,:,:]-combined[i,:,:])/8)/3).astype('uint16')
                interpolated[4*i+3,:,:]+=((combined[i,:,:]+3*(combined[i+1,:,:]-combined[i,:,:])/8)/3).astype('uint16')        
            print(i)
    else:
        for i in range(num_images):            
            if i==0:
                interpolated[2*i,:,:]+=((combined[i,:,:]-(combined[i+1,:,:]-combined[i,:,:])/4)/3).astype('uint16')       
            else:    
                interpolated[2*i,:,:]+=((combined[i,:,:]-(combined[i,:,:]-combined[i-1,:,:])/4)/3).astype('uint16')
    
            if i==(num_images-1):
                interpolated[2*i+1,:,:]+=((combined[i,:,:]+(combined[i,:,:]-combined[i-1,:,:])/4)/3).astype('uint16')
            else:    
                interpolated[2*i+1,:,:]+=((combined[i,:,:]+(combined[i+1,:,:]-combined[i,:,:])/4)/3).astype('uint16')       
            print(i)        
        
        
        
        
    res=images[0].split('-')[-1].split('_')[1].split('um')[0]
    new_res=str(float(images[0].split('-')[-1].split('_')[1].split('um')[0])/scale_factor)
    
    
    filename='SR'+'-'+images[0].split('-')[-2]+'-'+images[0].split('-')[-1].replace(res,new_res)     
        
    ###x slice
    img_dir=os.path.join(dirs,'x_slice')
    images=os.listdir(img_dir)
    num_images=len(images)
    dim=int(images[0].split('_')[-1].split('x')[0])

    combined=np.zeros((dim,dim,num_images)).astype(float)


    for i in range(num_images):
        image=utils.read_and_reshape(os.path.join(img_dir,images[i]))
        index=int(images[i].split('-')[1])
        combined[:,:,index]=image
        print(i)    
    
    if scale_factor==4:
        for i in range(num_images):
            if i==0:
                interpolated[:,:,4*i]+=((combined[:,:,i]-3*(combined[:,:,i+1]-combined[:,:,i])/8)/3).astype('uint16')
                interpolated[:,:,4*i+1]+=((combined[:,:,i]-(combined[:,:,i+1]-combined[:,:,i])/8)/3).astype('uint16')    
            else:    
                interpolated[:,:,4*i]+=((combined[:,:,i]-3*(combined[:,:,i]-combined[:,:,i-1])/8)/3).astype('uint16')
                interpolated[:,:,4*i+1]+=((combined[:,:,i]-(combined[:,:,i]-combined[:,:,i-1])/8)/3).astype('uint16')
    
            if i==(num_images-1):
                interpolated[:,:,4*i+2]+=((combined[:,:,i]+(combined[:,:,i]-combined[:,:,i-1])/8)/3).astype('uint16')
                interpolated[:,:,4*i+3]+=((combined[:,:,i]+3*(combined[:,:,i]-combined[:,:,i-1])/8)/3).astype('uint16')
            else:    
                interpolated[:,:,4*i+2]+=((combined[:,:,i]+(combined[:,:,i+1]-combined[:,:,i])/8)/3).astype('uint16')
                interpolated[:,:,4*i+3]+=((combined[:,:,i]+3*(combined[:,:,i+1]-combined[:,:,i])/8)/3).astype('uint16')        
            print(i)        
    else:
        for i in range(num_images):            
            if i==0:
                interpolated[:,:,2*i]+=((combined[:,:,i]-(combined[:,:,i+1]-combined[:,:,i])/4)/3).astype('uint16')       
            else:    
                interpolated[:,:,2*i]+=((combined[:,:,i]-(combined[:,:,i]-combined[:,:,i-1])/4)/3).astype('uint16')
    
            if i==(num_images-1):
                interpolated[:,:,2*i+1]+=((combined[:,:,i]+(combined[:,:,i]-combined[:,:,i-1])/4)/3).astype('uint16')
            else:    
                interpolated[:,:,2*i+1]+=((combined[:,:,i]+(combined[:,:,i+1]-combined[:,:,i])/4)/3).astype('uint16')       
            print(i)                
    ###y slice
    img_dir=os.path.join(dirs,'y_slice')
    images=os.listdir(img_dir)
    num_images=len(images)
    dim=int(images[0].split('_')[-1].split('x')[0])

    combined=np.zeros((dim,num_images,dim)).astype(float)

    for i in range(num_images):
        image=utils.read_and_reshape(os.path.join(img_dir,images[i]))
        index=int(images[i].split('-')[1])
        combined[:,index,:]=image
        print(i)
    if scale_factor==4:    
        for i in range(num_images):
            if i==0:
                interpolated[:,4*i,:]+=((combined[:,i,:]-3*(combined[:,i+1,:]-combined[:,i,:])/8)/3).astype('uint16')
                interpolated[:,4*i+1,:]+=((combined[:,i,:]-(combined[:,i+1,:]-combined[:,i,:])/8)/3).astype('uint16')       
            else:    
                interpolated[:,4*i,:]+=((combined[:,i,:]-3*(combined[:,i,:]-combined[:,i-1,:])/8)/3).astype('uint16')
                interpolated[:,4*i+1,:]+=((combined[:,i,:]-(combined[:,i,:]-combined[:,i-1,:])/8)/3).astype('uint16')
    
            if i==(num_images-1):
                interpolated[:,4*i+2,:]+=((combined[:,i,:]+(combined[:,i,:]-combined[:,i-1,:])/8)/3).astype('uint16')
                interpolated[:,4*i+3,:]+=((combined[:,i,:]+3*(combined[:,i,:]-combined[:,i-1,:])/8)/3).astype('uint16')
            else:    
                interpolated[:,4*i+2,:]+=((combined[:,i,:]+(combined[:,i+1,:]-combined[:,i,:])/8)/3).astype('uint16')
                interpolated[:,4*i+3,:]+=((combined[:,i,:]+3*(combined[:,i+1,:]-combined[:,i,:])/8)/3).astype('uint16')        
            print(i)
    else:
        for i in range(num_images):            
            if i==0:
                interpolated[:,2*i,:]+=((combined[:,i,:]-(combined[:,i+1,:]-combined[:,i,:])/4)/3).astype('uint16')       
            else:    
                interpolated[:,2*i,:]+=((combined[:,i,:]-(combined[:,i,:]-combined[:,i-1,:])/4)/3).astype('uint16')
    
            if i==(num_images-1):
                interpolated[:,2*i+1,:]+=((combined[:,i,:]+(combined[:,i,:]-combined[:,i-1,:])/4)/3).astype('uint16')
            else:    
                interpolated[:,2*i+1,:]+=((combined[:,i,:]+(combined[:,i+1,:]-combined[:,i,:])/4)/3).astype('uint16')       
            print(i)       

    res=images[0].split('-')[-1].split('_')[1].split('um')[0]
    new_res=str(float(images[0].split('-')[-1].split('_')[1].split('um')[0])/scale_factor)
    
    
    filename='SR'+'-'+images[0].split('-')[-2]+'-'+images[0].split('-')[-1].replace(res,new_res)
    utils.save_as_raw(interpolated,os.path.join(img_dir,filename),new_parent_dir='recons')