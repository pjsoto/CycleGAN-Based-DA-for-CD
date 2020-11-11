"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import sys
import torch
import importlib
import numpy as np
import torch.utils.data
from data.Amazonia_Legal_RO import AMAZON_RO
from data.Amazonia_Legal_PA import AMAZON_PA
from data.Cerrado_Biome_MA import CERRADO_MA
from data.base_dataset import BaseDataset
from sklearn.preprocessing import StandardScaler


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset

def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    scalers = []
    if opt.dataset_type == 'common_images':
        data_loader = CustomDatasetDataLoader(opt)
        dataset = data_loader.load_data()
        opt.linear_output = False
    if opt.dataset_type == 'remote_sensing_images':
        print('Loading the data...!')
        if opt.source_domain == 'Amazon_RO':
            opt.dataset = 'Amazonia_Legal/'
            opt.data_t1_name = opt.source_image_name_T1
            opt.data_t2_name = opt.source_image_name_T2
            dataset_s = AMAZON_RO(opt)
        if opt.source_domain == 'Amazon_PA':
            opt.dataset = 'Amazonia_Legal/'
            opt.data_t1_name = opt.source_image_name_T1
            opt.data_t2_name = opt.source_image_name_T2
            dataset_s = AMAZON_PA(opt)
        if opt.source_domain == 'Cerrado_MA':
            opt.dataset = 'Cerrado_Biome/'
            opt.data_t1_name = opt.source_image_name_T1
            opt.data_t2_name = opt.source_image_name_T2
            dataset_s = CERRADO_MA(opt)
                                    
        if opt.target_domain == 'Amazon_RO':
            opt.dataset = 'Amazonia_Legal/'
            opt.data_t1_name = opt.target_image_name_T1
            opt.data_t2_name = opt.target_image_name_T2
            dataset_t = AMAZON_RO(opt)
        if opt.target_domain == 'Amazon_PA':
            opt.dataset = 'Amazonia_Legal/'
            opt.data_t1_name = opt.target_image_name_T1
            opt.data_t2_name = opt.target_image_name_T2
            dataset_t = AMAZON_PA(opt)
        if opt.target_domain == 'Cerrado_MA':
            opt.dataset = 'Cerrado_Biome/'
            opt.data_t1_name = opt.target_image_name_T1
            opt.data_t2_name = opt.target_image_name_T2
            dataset_t = CERRADO_MA(opt)
        
        print(np.shape(dataset_s.images_norm))
        print(np.shape(dataset_t.images_norm))   
                
        if opt.standard_normalization:
            opt.convert = False
            opt.linear_output = True
        else:
            opt.convert = True
            opt.linear_output = False
        
        scalers.append(dataset_s.scaler[0])
        scalers.append(dataset_t.scaler[0])
        scalers.append(dataset_s.scaler[1])
        scalers.append(dataset_t.scaler[1])
                
        opt.rows_size_s = np.size(dataset_s.images_norm[0], 0)
        opt.cols_size_s = np.size(dataset_s.images_norm[0], 1)
        opt.rows_size_t = np.size(dataset_t.images_norm[0], 0)
        opt.cols_size_t = np.size(dataset_t.images_norm[0], 1)
           
        data_loader = RemoteSensingCustomDatasetDataLoader(dataset_s, dataset_t, opt)
        dataset = data_loader.load_data()
        
    return dataset, scalers

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print(self.dataset)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

class RemoteSensingCustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading in remote sensing data"""

    def __init__(self, dataset_s, dataset_t, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        if self.opt.phase == 'train':
            mask_s = np.ones((dataset_s.images_norm[0].shape[0], dataset_s.images_norm[0].shape[1]))
            patch_coordinates_s,_,_,_ = self.Central_Pixel_Definition(mask_s, np.zeros((dataset_s.images_norm[0].shape[0], dataset_s.images_norm[0].shape[1])), np.zeros((dataset_s.images_norm[0].shape[0], dataset_s.images_norm[0].shape[1])), self.opt.crop_size, self.opt.stride_s, 100)
            mask_t = np.ones((dataset_t.images_norm[0].shape[0], dataset_t.images_norm[0].shape[1]))
            patch_coordinates_t,_,_,_ = self.Central_Pixel_Definition(mask_t, np.zeros((dataset_t.images_norm[0].shape[0], dataset_t.images_norm[0].shape[1])), np.zeros((dataset_t.images_norm[0].shape[0], dataset_t.images_norm[0].shape[1])), self.opt.crop_size, self.opt.stride_t, 100)
            
            size_s = patch_coordinates_s.shape[0]
            size_t = patch_coordinates_t.shape[0]
            # Taking equal number of samples for both domains          
            if size_s < size_t:
                diff = size_t - size_s 
                patch_coordinates_s = np.concatenate((patch_coordinates_s , patch_coordinates_s[:diff,:]),axis=0)
            if size_t < size_s:
                diff = size_s - size_t 
                patch_coordinates_t = np.concatenate((patch_coordinates_t , patch_coordinates_t[:diff,:]),axis=0)
            
            
            
        if self.opt.phase == 'test':
            patch_coordinates_s = self.Coordinates_Definition_Test(dataset_s.images_norm[0].shape[0], dataset_s.images_norm[0].shape[1], self.opt, True)
            patch_coordinates_t = self.Coordinates_Definition_Test(dataset_t.images_norm[0].shape[0], dataset_t.images_norm[0].shape[1], self.opt, False)        
            
            opt.size_s = patch_coordinates_s.shape[0]
            opt.size_t = patch_coordinates_t.shape[0]
            # Taking equal number of samples for both domains          
            if opt.size_s < opt.size_t:
                diff = opt.size_t - opt.size_s 
                patch_coordinates_s = np.concatenate((patch_coordinates_s , patch_coordinates_s[:diff,:]),axis=0)
            if opt.size_t < opt.size_s:
                diff = opt.size_s - opt.size_t 
                patch_coordinates_t = np.concatenate((patch_coordinates_t , patch_coordinates_t[:diff,:]),axis=0)
            
             
        if self.opt.same_coordinates:
            # Here the coordinates will be the same for both pair of images, i.e., paired samples will be produced
            dataset_s.central_pixels_coordinates = patch_coordinates_s
            dataset_t.central_pixels_coordinates = patch_coordinates_t
            self.dataset = dataset_class(dataset_s, dataset_t, opt) 
        else:
            # Performing a shuffle operation in order to get different pairs of images
            num_samples_s = patch_coordinates_s.shape[0]
            num_samples_t = patch_coordinates_t.shape[0]
            indexs_s = np.arange(num_samples_s)
            indexs_t = np.arange(num_samples_t)
            np.random.shuffle(indexs_s)
            np.random.shuffle(indexs_t)
            dataset_s.central_pixels_coordinates = patch_coordinates_s[indexs_s, :]
            dataset_t.central_pixels_coordinates = patch_coordinates_t[indexs_t, :]
            self.dataset = dataset_class(dataset_s, dataset_t, opt)  
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        print('pedro')
        for i, data in enumerate(self.dataloader):
            
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
    
    def Central_Pixel_Definition(self, mask, last_reference, actual_reference, patch_dimension, stride, porcent_of_last_reference_in_actual_reference):
        # This method will select the initial coordinates based on central pixel critetion. This just will be applied
        # for training step. In test, a differente procedure is used to extract the patches.
        mask_rows = np.size(mask, 0)
        mask_cols = np.size(mask, 1)
        
        half_dim = patch_dimension//2
        upper_padding = np.zeros((half_dim, mask_cols))
        left_padding = np.zeros((mask_rows + half_dim, half_dim))
        bottom_padding = np.zeros((half_dim, half_dim + mask_cols))
        right_padding = np.zeros((2 * half_dim + mask_rows, half_dim))
        
        #Add padding to the mask     
        mask_padded = np.concatenate((upper_padding, mask), axis=0)
        mask_padded = np.concatenate((left_padding, mask_padded), axis=1)
        mask_padded = np.concatenate((mask_padded, bottom_padding), axis=0)
        mask_padded = np.concatenate((mask_padded, right_padding), axis=1)
        
        #Add padding to the last reference
        last_reference_padded = np.concatenate((upper_padding, last_reference), axis=0)
        last_reference_padded = np.concatenate((left_padding, last_reference_padded), axis=1)
        last_reference_padded = np.concatenate((last_reference_padded, bottom_padding), axis=0)
        last_reference_padded = np.concatenate((last_reference_padded, right_padding), axis=1)
        
        #Add padding to the last reference
        actual_reference_padded = np.concatenate((upper_padding, actual_reference), axis=0)
        actual_reference_padded = np.concatenate((left_padding, actual_reference_padded), axis=1)
        actual_reference_padded = np.concatenate((actual_reference_padded, bottom_padding), axis=0)
        actual_reference_padded = np.concatenate((actual_reference_padded, right_padding), axis=1)
        
        #Initializing the central pixels coordinates containers
        central_pixels_coord_tr_init = []
        central_pixels_coord_vl_init = []
        
        if stride == 1:
            central_pixels_coord_tr_init = np.where(mask_padded == 1)
            central_pixels_coord_vl_init = np.where(mask_padded == 3)
            central_pixels_coord_tr_init = np.transpose(np.array(central_pixels_coord_tr_init))
            central_pixels_coord_vl_init = np.transpose(np.array(central_pixels_coord_vl_init))
        else:
            counter_tr = 0
            counter_vl = 0
            for i in range(2 * half_dim, np.size(mask_padded , 0) - 2 * half_dim, stride):
                for j in range(2 * half_dim, np.size(mask_padded , 1) - 2 * half_dim, stride):
                    mask_value = mask_padded[i , j]
                    #print(mask_value)
                    if mask_value == 1:
                        #Belongs to the training tile
                        counter_tr += 1
                        
                    if mask_value == 3:
                        #Belongs to the validation tile
                        counter_vl += 1
            
            central_pixels_coord_tr_init = np.zeros((counter_tr, 2))
            central_pixels_coord_vl_init = np.zeros((counter_vl, 2))
            counter_tr = 0
            counter_vl = 0        
            for i in range(2 * half_dim , np.size(mask_padded , 0) - 2 * half_dim, stride):
                for j in range(2 * half_dim , np.size(mask_padded , 1) - 2 * half_dim, stride):
                    mask_value = mask_padded[i , j]
                    #print(mask_value)
                    if mask_value == 1:
                        #Belongs to the training tile
                        central_pixels_coord_tr_init[counter_tr , 0] = int(i)
                        central_pixels_coord_tr_init[counter_tr , 1] = int(j)
                        counter_tr += 1                    
                    if mask_value == 3:
                        #Belongs to the validation tile
                        central_pixels_coord_vl_init[counter_vl , 0] = int(i)
                        central_pixels_coord_vl_init[counter_vl , 1] = int(j)
                        counter_vl += 1
        
        #Refine the central pixels coordinates
        counter_tr = 0
        counter_vl = 0
        for i in range(np.size(central_pixels_coord_tr_init , 0)):
            last_reference_value = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
            actual_reference_value = actual_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
            if (last_reference_value != 1) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) - half_dim : int(central_pixels_coord_tr_init[i, 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i, 1]) - half_dim : int(central_pixels_coord_tr_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    counter_tr += 1
        
        for i in range(np.size(central_pixels_coord_vl_init , 0)):
            last_reference_value = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
            actual_reference_value = actual_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
            if (last_reference_value != 1 ) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) - half_dim : int(central_pixels_coord_vl_init[i, 0]) + half_dim  + 1, int(central_pixels_coord_vl_init[i, 1]) - half_dim : int(central_pixels_coord_vl_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    counter_vl += 1
                
        central_pixels_coord_tr = np.zeros((counter_tr, 2))
        central_pixels_coord_vl = np.zeros((counter_vl, 2))
        y_train_init = np.zeros((counter_tr,1))
        y_valid_init = np.zeros((counter_vl,1))
        counter_tr = 0
        counter_vl = 0
        for i in range(np.size(central_pixels_coord_tr_init , 0)):
            last_reference_value = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
            actual_reference_value = actual_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
            if (last_reference_value != 1 ) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) - half_dim : int(central_pixels_coord_tr_init[i, 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i, 1]) - half_dim : int(central_pixels_coord_tr_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    central_pixels_coord_tr[counter_tr, 0] = central_pixels_coord_tr_init[i , 0]
                    central_pixels_coord_tr[counter_tr, 1] = central_pixels_coord_tr_init[i , 1]
                    y_train_init[counter_tr, 0] = actual_reference_value
                    counter_tr += 1
                
        for i in range(np.size(central_pixels_coord_vl_init , 0)):
            last_reference_value = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
            actual_reference_value = actual_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
            if (last_reference_value != 1 ) and (actual_reference_value <= 1):
                last_reference_patch = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) - half_dim : int(central_pixels_coord_vl_init[i, 0]) + half_dim + 1, int(central_pixels_coord_vl_init[i, 1]) - half_dim : int(central_pixels_coord_vl_init[i, 1]) + half_dim + 1]
                number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
                number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
                porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
                if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                    central_pixels_coord_vl[counter_vl, 0] = central_pixels_coord_vl_init[i , 0]
                    central_pixels_coord_vl[counter_vl, 1] = central_pixels_coord_vl_init[i , 1]
                    y_valid_init[counter_vl, 0] = actual_reference_value
                    counter_vl += 1
    
        return central_pixels_coord_tr, y_train_init, central_pixels_coord_vl, y_valid_init
    
    def Coordinates_Definition_Test(self, rows_size, cols_size, opt, from_source):
        
        if from_source:
            overlap = round(opt.crop_size * opt.overlap_porcent_s)
        else:
            overlap = round(opt.crop_size * opt.overlap_porcent_t)
        
        overlap -= overlap % 2
        stride = opt.crop_size - overlap
        
        step_row = (stride - rows_size % stride) % stride
        step_col = (stride - cols_size % stride) % stride
        
        k1, k2 = (rows_size + step_row)//stride, (cols_size + step_col)//stride
        coordinates = np.zeros((k1 * k2 , 4))
        counter = 0
        for i in range(k1):
            for j in range(k2):
                coordinates[counter, 0] = i * stride
                coordinates[counter, 1] = j * stride
                coordinates[counter, 2] = i * stride + opt.crop_size
                coordinates[counter, 3] = j * stride + opt.crop_size
                counter += 1
        
        return coordinates
                
         
            
    
       