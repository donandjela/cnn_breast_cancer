
# coding: utf-8

# In[1]:


from PIL import Image
from random import  shuffle
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import  train_test_split
import Data_augmentation


# In[2]:


def get_train_data_names(path: str , disp: bool = False, encoded:bool = True):
    """ 
    Arguments:
        path: path to the specific folder
        disp: possible values {True, False}. Default value is False. If True function displays
                generated DataFrames
        encoded: possible values {True, False}. Default value i True. If True function encodes target label
                 using pd.get_dummies
                
    Returns:
        info: DataFrame with information about train data (labels, initial data count, augmented data count)
        train_data: DataFrame with names of train data files and labels given as (rowvise):
                    ['file_1.tif', label_1]
                    ['file_2'.tiff, label_2]
                    .
                    .
                    .
                    ['file_n.tif', label_n]
                    
                    if encoded == True function returns DataFrame given as:
                    ['file_1.tif', label_1 encoded]
                    ['file_2'.tiff, label_2 encoded]
                    .
                    .
                    .
                    ['file_n.tif', label_n encoded]
        
        separated_classes_data: Dict of labels and names of train data files given as:
                    {0 : benign_data, 
                     1: insitu_data,
                     2: invasive_data,
                     3: normal_data}
    """
    

    legend = {'classes' : ['Benign', 'InSitu','Invasive','Normal'], 'labels': [0,1,2,3]}
    info = pd.DataFrame(legend,columns=['classes','labels'])

    path_benign = os.path.join(path, 'Benign')
    data = os.walk(path_benign)
    names = [d for d in data][0][2]
    benign_data = [(os.path.join(path_benign,name),0) for name in names]

    path_insitu = os.path.join(path,'InSitu')
    data = os.walk(path_insitu)
    names = [d for d in data][0][2]
    insitu_data = [(os.path.join(path_insitu, name),1) for name in names]
    
    path_invasive = os.path.join(path,'Invasive')
    data = os.walk(path_invasive)
    names = [d for d in data][0][2]
    invasive_data = [(os.path.join(path_invasive,name),2) for name in names]
    
    path_normal = os.path.join(path, 'Normal')
    data = os.walk(path_normal)
    names = [d for d in data][0][2]
    normal_data = [(os.path.join(path_normal,name),3) for name in names]
    
    train_data = pd.DataFrame(benign_data + insitu_data + invasive_data+ normal_data,
                              columns=['Name', 'Target label'])
    

    augmented_train_count = [len(benign_data),
                             len(insitu_data),
                             len(invasive_data),
                             len(normal_data)]
    
    info['Initial train count'] =  [i/(35*8) for i in augmented_train_count]
    info['Augmented train count'] = augmented_train_count
    
    
    if disp:
        display(info)
        
        
    separated_classes_data = {0 : benign_data, 
                             1: insitu_data,
                             2: invasive_data,
                             3: normal_data}
    
    if encoded:
        return info, encode_target(train_data), separated_classes_data
    
    return info, train_data, separated_classes_data


# In[3]:


def get_test_data(path:str, disp: bool = False, encoded: bool = True):
    
    """ 
    Arguments:
        path: path to the specific folder
        disp: posiible values {True, False}. Default value is False. If True function displays
                generated DataFrames
        encoded: possible values {True, False}. Default value i True. If True function encodes target label
                 using pd.get_dummies
                
    Returns:
        info: DataFrame with information about test data (labels, initial data count, augmented data count)
        test_data: DataFrame with names of test data files and labels given as (rowvise):
                    ['file_1.tif', label_1]
                    ['file_2'.tiff, label_2]
                    .
                    .
                    .
                    ['file_n.tif', label_n]
                    
                if encoded == True function returns DataFrame given as:
                    ['file_1.tif', label_1 encoded]
                    ['file_2'.tiff, label_2 encoded]
                    .
                    .
                    .
                    ['file_n.tif', label_n encoded]
        
    """
    

    legend = {'classes' : ['Benign', 'InSitu','Invasive','Normal'], 'labels': [0,1,2,3]}
    info = pd.DataFrame(legend,columns=['classes','labels'])
    
    test_data_info = pd.read_csv(os.path.join(path, 'labels.txt'),delimiter=' ')
    
    test_data_info['Target label'] = test_data_info.Type.replace(legend['classes'], legend['labels'])

    test_data = test_data_info[['Name', 'Target label']]
    
    names = [os.path.join(path, 't' + name) for name in test_data.Name.values]
    
    test_data['Name'] = names
    
    info['Initial test count'] = [i/(12*8) for i in test_data_info.groupby(['Type']).count().Name.values]
    info['Augmented test count'] = test_data_info.groupby(['Type']).count().Name.values
    
    if disp:
        display(info)
        
    if encoded:
        return info, encode_target(test_data)
    
    return info, test_data


# In[4]:


def get_val_data_names(path: str , disp: bool = False, encoded:bool = True):
    """ 
    Arguments:
        path: path to the specific folder
        disp: possible values {True, False}. Default value is False. If True function displays
                generated DataFrames
        encoded: possible values {True, False}. Default value i True. If True function encodes target label
                 using pd.get_dummies
                
    Returns:
        info: DataFrame with information about validation data (labels, initial data count, augmented data count)
        train_data: DataFrame with names of train data files and labels given as (rowvise):
                    ['file_1.tif', label_1]
                    ['file_2'.tiff, label_2]
                    .
                    .
                    .
                    ['file_n.tif', label_n]
                    
                    if encoded == True function returns DataFrame given as:
                    ['file_1.tif', label_1 encoded]
                    ['file_2'.tiff, label_2 encoded]
                    .
                    .
                    .
                    ['file_n.tif', label_n encoded]
        
        separated_classes_data: Dict of labels and names of train data files given as:
                    {0 : benign_data, 
                     1: insitu_data,
                     2: invasive_data,
                     3: normal_data}
    """
    

    legend = {'classes' : ['Benign', 'InSitu','Invasive','Normal'], 'labels': [0,1,2,3]}
    info = pd.DataFrame(legend,columns=['classes','labels'])

    path_benign = os.path.join(path, 'Benign')
    data = os.walk(path_benign)
    names = [d for d in data][0][2]
    benign_data = [(os.path.join(path_benign,name),0) for name in names]

    path_insitu = os.path.join(path,'InSitu')
    data = os.walk(path_insitu)
    names = [d for d in data][0][2]
    insitu_data = [(os.path.join(path_insitu, name),1) for name in names]
    
    path_invasive = os.path.join(path,'Invasive')
    data = os.walk(path_invasive)
    names = [d for d in data][0][2]
    invasive_data = [(os.path.join(path_invasive,name),2) for name in names]
    
    path_normal = os.path.join(path, 'Normal')
    data = os.walk(path_normal)
    names = [d for d in data][0][2]
    normal_data = [(os.path.join(path_normal,name),3) for name in names]
    
    train_data = pd.DataFrame(benign_data + insitu_data + invasive_data+ normal_data,
                              columns=['Name', 'Target label'])
    

    augmented_train_count = [len(benign_data),
                             len(insitu_data),
                             len(invasive_data),
                             len(normal_data)]
    
    info['Initial val count'] =  [i/(35*8) for i in augmented_train_count]
    info['Augmented val count'] = augmented_train_count
    
    
    if disp:
        display(info)
        #display(train_data.head(10))
        
    separated_classes_data = {0 : benign_data, 
                             1: insitu_data,
                             2: invasive_data,
                             3: normal_data}
    
    if encoded:
        return info, encode_target(train_data), separated_classes_data
    
    return info, train_data, separated_classes_data


# In[5]:


def encode_target(data: pd.DataFrame, col ='Target label' ):
    
    return pd.concat([data.copy(), pd.get_dummies(data[col])], axis=1).drop(columns= [col])
    


# In[6]:


class BatchGenerator():
    #(keras.utils.Sequence):
    
    """
    BatchGenerator : Generator class used to yield the next training batch.
    """
    
    def __init__(self,data_samples : pd.DataFrame, batch_size : int = 32):
        
        """
        BatchGenerator init
        Arguments:
            data_samples: DataFrame with names of data files and labels given as (rowvise):
                    ['file_1.tif', label_1 encoded]
                    ['file_2'.tiff, label_2 encoded]
                    .
                    .
                    .
                    ['file_n.tif', label_n encoded]
        
        batch_size: size of a batch, default value is 32
        
        """
        
        self.samples = list(data_samples.values)
        self.batch_size = batch_size
    
    def generate(self):
        """
        Generator function which yields data
        
        Arguments:
            no arguments
        Yields:
            X_train: List of images given as np.arrays
            y_trian: list of corresponding encoded labels
        
        
        """
        
        samples = self.samples
        batch_size = self.batch_size
        
        num_samples = len(samples)
        
        while True:
            shuffle(samples)
            
            for i in range(0, num_samples, batch_size):
                
                batch_samples = samples[i:i+batch_size]
                
                X_train = np.zeros((batch_size,512,512,3))
                y_train = np.zeros((batch_size,4))
                
                for i, batch_sample in enumerate(batch_samples):
                    
                    filename = batch_sample[0]
                    im = Image.open(filename)
                    image = np.array(im)/255.0
                    
                    y = batch_sample[1:]
                    
                    X_train[i] = image
                    y_train[i] = y
                
                yield X_train, y_train


# In[10]:




