
# coding: utf-8

# In[1]:


import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def py_patch(img : np.array, patch_size: 'tuple of ints' = (512,512,3), disp: bool = False, overlap: bool = False):
    
    """
    Divides image into patches
    
    Arguments: 
        img: image given as np.array
        patch_size: size of a single patch
        disp: possible values {True, False}, default value is False. If set to True function displays patches
              as images
        overlap: possible values {True, False}, default value is False. Defines whether generated patches will
                 have 50% overlap. If overlap == True patches have overlap, if overlap = False  there is no overlap
                 between patches
    
    Returns:
        If overlap == True it returns overlaping pathches with 50% overlap. Otherwise function returns
        nonoverlaping patches
        
    """
    
    if overlap == False:
        return py_patch_nonoverlaping(img, patch_size = patch_size, disp = disp)
    else:
        return py_patch_overlaping(img, patch_size = patch_size, disp = disp)


# In[3]:


# Custom patching with no overlap
def py_patch_nonoverlaping(img: np.array , patch_size: tuple =(512,512,3), disp:bool = False):
    
    """
    Diviges image into nonoverlaping patches
    Arguments:
        img: image given as np.array
        patch_size: size of a single patch
        disp: possible values {True, False}, default value is False. If set to True function displays patches
              as images
              
    Returns: 
        patches: list of nonoverlaping patches given as np.arrays
        nb_x: number of patches from x-axis
        nb_y: number of patches from y-axis
    """

    patch_x = patch_size[0]
    patch_y = patch_size[1]
    
    h,w,_ = img.shape

    nb_x = int(h/patch_x)
    nb_y = int(w/patch_y)
    
    patches = []
    
    for i in range(nb_x):
        from_x = int(i*patch_x)
        to_x = int((i+1)*patch_x)
        
        for j in range(nb_y):
            from_y = int(j*patch_y)
            to_y = int((j+1)*patch_y)
            patches.append(img[from_x :to_x ,from_y:to_y,:])
    
    if disp==True:
        show_images(patches)
    
    
    return patches, nb_x, nb_y
    
    


# In[4]:


# Custom patching with 50% overlap

def py_patch_overlaping(img: np.array, patch_size: tuple = (512,512,3), disp: bool = False):
    
    """
    Diviges image into overlaping patches with 50% overlap
    Arguments:
        img: image given as np.array
        patch_size: size of a single patch
        disp: possible values {True, False}, default value is False. If set to True function displays patches
              as images
              
    Returns: 
        patches: list of overlaping patches given as np.arrays
        nb_x: number of patches from x-axis
        nb_y: number of patches from y-axis
    
    """
  
    patch_x = patch_size[0]
    patch_y = patch_size[1]
    
    h,w,_ = img.shape

    nb_x = int(2*h/patch_x)-1
    nb_y = int(2*w/patch_y)-1
    
    patches = []
    
    for i in range(nb_x):
        from_x = int(i*patch_x/2)
        to_x = int(i*patch_x/2 + patch_x)
        
        for j in range(nb_y):
            from_y = int(j*patch_y/2)
            to_y = int(j*patch_y/2 + patch_y)
            patches.append(img[from_x :to_x ,from_y:to_y,:])
            
    if disp==True:
        show_images(patches)
            
    return patches, nb_x, nb_y


# In[5]:



def show_images(img_list: 'list of np.arrays', nb_cols: int = 4):
    
    """
    Displays list of images
    
    Arguments:
        img_list: list of images given as np.array
        nb_cols: number of columns in desired representation
        
    """
    
    
    nb_imgs = len(img_list)
    nb_rows, nb_cols_last_row = divmod(nb_imgs, nb_cols)
   
    k = 0
    
    if nb_cols_last_row ==0:
        fig, ax = plt.subplots(nb_rows,nb_cols, figsize=(20,nb_rows*5),
                              subplot_kw ={'xticks':(), 'yticks':()})
    else:
        fig, ax = plt.subplots(nb_rows+1,nb_cols, figsize=(15,nb_rows*5),
                              subplot_kw ={'xticks':(), 'yticks':()} )
    for i in range(nb_rows):
        for j in range(nb_cols):

                ax[i,j].set_title('Image ' + str(k+1), fontsize = 10)
                ax[i,j].imshow(Image.fromarray(img_list[k], 'RGB'))
                k+=1    
                if k==(nb_imgs):
                    break
                
    if (nb_cols_last_row !=0):
        
        for i in range(nb_cols_last_row):
            k=nb_rows*nb_cols+i
            ax[nb_rows, i].imshow(Image.fromarray(img_list[k], 'RGB'))
            ax[nb_rows, i].set_title('Image ' + str(k+1), fontsize = 10)
            
        for i in range(nb_cols - nb_cols_last_row):
            ax[nb_rows, nb_cols_last_row + i].set_axis_off()
            


# In[6]:


#reflection of an image
def mirror(img: Image):
    
    """
    Reflects an image
    Arguments:
        img: image given as Image object
        
    Returns:
        reflected image
    
    """
    
    return np.array(img.transpose(Image.FLIP_LEFT_RIGHT))


# In[7]:


#4 image rotations
def rotate(img: Image)-> 'list of np.arrays':
    """
    Rotates an image for 0,90,180,360 degrees
    
    Arguments:
        img: image given as Image object
    
    Returns
        rotated_imgs: list of rotated images representes as np.arrays
    
    """
    rotated_imgs = []
    angles = [k* 90 for k in [0,1,2,3]]
    
    for angle in angles:
        
        rotated_imgs.append(np.array(img.rotate(angle)))
        
    return rotated_imgs


# In[8]:


# transformation of an image with mirroring and 4 rotations
# returns a np.array of patches given also as np.array-s
def transform(img: np.array) -> 'list of np.arrays':
    
    """
    Transforms an image with respect to mirroring and 4 rotations for 0,90,180,360 degrees
    
    Arguments:
        img: image given as np.array
        
    Returns:
        list of transformed images
    
    """
    
    img = Image.fromarray(img,'RGB')
    mirror_img = mirror(img)
    
    return rotate(img) + (rotate(Image.fromarray(mirror_img,'RGB')))


# In[9]:


# TEST
#img_1 = np.array(Image.open('t0.tif'))
#img_2 = Image.open('example2.tif')

#A = py_patch(img_1, patch_size = (512,512,3), disp = True, overlap = True)

