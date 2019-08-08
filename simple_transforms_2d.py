import PIL
from PIL import Image
import numpy as np



def resize2d(img,size,resample='BICUBIC'):
    '''
    Takes in a 2d image file and resize to desired size, several resample method can be applied. This function is based on PIL library in python.
    
    Input:  
    img: numpy array of 2D image
    size: requested size as 2-tuple
    resample: resampling filter to be used
    
    Return:
    new numpy array of resized image
    
    '''
    im=Image.fromarray(img.astype(float))
    if resample=='NEAREST':
        new_im=im.resize(size,resample=PIL.Image.NEAREST)
    elif resample=='BILINEAR':
        new_im=im.resize(size,resample=PIL.Image.BILINEAR)
    elif resample=='BICUBIC':
        new_im=im.resize(size,resample=PIL.Image.BICUBIC)    
    elif resample=='LANCZOS':
        new_im=im.resize(size,resample=PIL.Image.LANCZOS)
    
    np_im=np.array(new_im)
    return np_im

