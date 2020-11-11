import os
import sys
import numpy as np
import skimage.morphology
from skimage.morphology import square, disk 
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class AMAZON_RO():
    def __init__(self, args):
        
        self.images_norm = []
        self.scaler = []
        self.images_diff = []
        
        Image_t1_path = args.dataroot + args.dataset + args.images_section + args.data_t1_name + '.npy'
        Image_t2_path = args.dataroot + args.dataset + args.images_section + args.data_t2_name + '.npy'
        
        # Reading images and references
        print('[*]Reading images...')
        image_t1 = np.load(Image_t1_path)
        image_t2 = np.load(Image_t2_path)
        
        
        #Cutting the last rows and columns of the images
        image_t1 = image_t1[:,1:2551,1:5121]
        image_t2 = image_t2[:,1:2551,1:5121]
        
        
        # Pre-processing images
        if args.compute_ndvi:
            print('[*]Computing and stacking the ndvi band...')            
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
            image_t1 = np.concatenate((image_t1, ndvi_t1), axis=2)
            image_t2 = np.concatenate((image_t2, ndvi_t2), axis=2)
        else:
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
        # Computing the difference image
        image_di = image_t2 - image_t1       
        # Pre-Processing the images
        if args.standard_normalization:
            print('[*]Normalizing the images...')
            scaler = StandardScaler()

            images = np.concatenate((image_t1, image_t2), axis=2)
            images_reshaped = images.reshape((images.shape[0] * images.shape[1], images.shape[2]))
            
            scaler = scaler.fit(images_reshaped)
            self.scaler.append(scaler)
            images_normalized = scaler.fit_transform(images_reshaped)
            images_norm = images_normalized.reshape((images.shape[0], images.shape[1], images.shape[2]))
            image_t1_norm = images_norm[:, :, : image_t1.shape[2]]
            image_t2_norm = images_norm[:, :, image_t2.shape[2]: ]
            
            # Storing the images in a list
            self.images_norm.append(image_t1_norm)
            self.images_norm.append(image_t2_norm)
            scaler = StandardScaler()
            images_reshaped = image_di.reshape((image_di.shape[0] * image_di.shape[1], image_di.shape[2]))
            scaler = scaler.fit(images_reshaped)
            self.scaler.append(scaler)
            images_normalized = scaler.fit_transform(images_reshaped)
            image_di_norm = images_normalized.reshape((image_di.shape[0], image_di.shape[1], image_di.shape[2]))
            self.images_diff.append(image_di_norm)