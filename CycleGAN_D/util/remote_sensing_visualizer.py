import numpy as np
import scipy.io as sio
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError




class RemoteSensingVisualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, scalers, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.scalers = scalers # remote sensing data normalizers
        self.display_id = opt.display_id
        #self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        # if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
        self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        #if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
        #    self.saved = True
            # save images to the disk
        for label, image in visuals.items():
            image_tensor = image.data
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = np.transpose(image_numpy,(1, 2, 0))
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.mat' % (epoch, label))
            image_reshaped = image_numpy.reshape((image_numpy.shape[0] * image_numpy.shape[1], image_numpy.shape[2]))
            if 'A' in label:
                if 'diff' not in label:
                    # Taking back the scalers of the T1 combination
                    scaler_1 = self.scalers[0]
                    image_inv = scaler_1.inverse_transform(image_reshaped)
                if 'diff' in label:
                    scaler_3 = self.scalers[2]
                    image_inv = scaler_3.inverse_transform(image_reshaped)
            if 'B' in label:
                if 'diff' not in label:
                    # Taking back the scalers of the T2 combination
                    scaler_2 = self.scalers[1]
                    image_inv = scaler_2.inverse_transform(image_reshaped)
                if 'diff' in label:
                    scaler_4 = self.scalers[3]
                    image_inv = scaler_4.inverse_transform(image_reshaped)
            
            image = image_inv.reshape((image_numpy.shape[0], image_numpy.shape[1], image_numpy.shape[2]))
            sio.savemat(img_path, {label: image})

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


class RemoteSensingPatchesContainer():
    
    def __init__(self, scalers, opt):
        
        self.opt = opt
        self.scalers = scalers
        self.save_path = self.opt.results_dir + self.opt.name + '/images/'
        # Creating the directories
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path) 
        #Computing the coordinates and prepare the container for the patches
        self.overlap_s = round(self.opt.crop_size * self.opt.overlap_porcent_s)
        self.overlap_s -= self.overlap_s % 2
        self.stride_s = self.opt.crop_size - self.overlap_s
        
        self.overlap_t = round(self.opt.crop_size * self.opt.overlap_porcent_t)
        self.overlap_t -= self.overlap_t % 2
        self.stride_t = self.opt.crop_size - self.overlap_t
        
        self.step_row_s = (self.stride_s - self.opt.rows_size_s % self.stride_s) % self.stride_s
        self.step_col_s = (self.stride_s - self.opt.cols_size_s % self.stride_s) % self.stride_s
        
        self.step_row_t = (self.stride_t - self.opt.rows_size_t % self.stride_t) % self.stride_t
        self.step_col_t = (self.stride_t - self.opt.cols_size_t % self.stride_t) % self.stride_t
        
        
        self.k1_s, self.k2_s = (self.opt.rows_size_s + self.step_row_s)//self.stride_s, (self.opt.cols_size_s + self.step_col_s)//self.stride_s
        self.coordinates_s = np.zeros((self.k1_s * self.k2_s , 4))
        
        self.k1_t, self.k2_t = (self.opt.rows_size_t + self.step_row_t)//self.stride_t, (self.opt.cols_size_t + self.step_col_t)//self.stride_t
        self.coordinates_t = np.zeros((self.k1_t * self.k2_t , 4))
        
        if self.opt.save_real:
            self.patchcontainer_real_A = np.zeros((self.k1_s * self.stride_s, self.k2_s * self.stride_s, self.opt.output_nc))
            self.patchcontainer_real_B = np.zeros((self.k1_t * self.stride_t, self.k2_t * self.stride_t, self.opt.output_nc))
        
        self.patchcontainer_fake_A = np.zeros((self.k1_t * self.stride_t, self.k2_t * self.stride_t, self.opt.output_nc))
        self.patchcontainer_fake_B = np.zeros((self.k1_s * self.stride_s, self.k2_s * self.stride_s, self.opt.output_nc))
        
        counter = 0
        for i in range(self.k1_s):
            for j in range(self.k2_s):
                self.coordinates_s[counter, 0] = i * self.stride_s
                self.coordinates_s[counter, 1] = j * self.stride_s
                self.coordinates_s[counter, 2] = i * self.stride_s + self.opt.crop_size
                self.coordinates_s[counter, 3] = j * self.stride_s + self.opt.crop_size
                counter += 1
                
        counter = 0
        for i in range(self.k1_t):
            for j in range(self.k2_t):
                self.coordinates_t[counter, 0] = i * self.stride_t
                self.coordinates_t[counter, 1] = j * self.stride_t
                self.coordinates_t[counter, 2] = i * self.stride_t + self.opt.crop_size
                self.coordinates_t[counter, 3] = j * self.stride_t + self.opt.crop_size
                counter += 1
    
    def store_current_visuals(self, visuals, index):

        for label, image in visuals.items():
            image_tensor = image.data
            image_numpy = image_tensor[0].cpu().float().numpy()
            image_numpy = np.transpose(image_numpy,(1, 2, 0))
            
            if label == 'fake_A':
                if index < self.opt.size_t:
                    self.patchcontainer_fake_A[int(self.coordinates_t[index, 0]) : int(self.coordinates_t[index, 0]) + int(self.stride_t), 
                                            int(self.coordinates_t[index, 1]) : int(self.coordinates_t[index, 1]) + int(self.stride_t), :] = image_numpy[int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),
                                                                                                                                    int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),:]
            if label == 'fake_B':
                if index < self.opt.size_s:
                    self.patchcontainer_fake_B[int(self.coordinates_s[index, 0]) : int(self.coordinates_s[index, 0]) + int(self.stride_s), 
                                            int(self.coordinates_s[index, 1]) : int(self.coordinates_s[index, 1]) + int(self.stride_s), :] = image_numpy[int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),
                                                                                                                                    int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),:]
            if self.opt.save_real:
                if label == 'real_A':
                    if index < self.opt.size_s:
                        self.patchcontainer_real_A[int(self.coordinates_s[index, 0]) : int(self.coordinates_s[index, 0]) + int(self.stride_s), 
                                            int(self.coordinates_s[index, 1]) : int(self.coordinates_s[index, 1]) + int(self.stride_s), :] = image_numpy[int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),
                                                                                                                                    int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),:]
                if label == 'real_B':
                    if index < self.opt.size_t:
                        self.patchcontainer_real_B[int(self.coordinates_t[index, 0]) : int(self.coordinates_t[index, 0]) + int(self.stride_t), 
                                            int(self.coordinates_t[index, 1]) : int(self.coordinates_t[index, 1]) + int(self.stride_t), :] = image_numpy[int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),
                                                                                                                                    int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),:]
                
    def save_images(self):
        
        fake_img_A = self.patchcontainer_fake_A[:self.k1_t*self.stride_t - self.step_row_t, :self.k2_t*self.stride_t - self.step_col_t, :]
        fake_img_B = self.patchcontainer_fake_B[:self.k1_s*self.stride_s - self.step_row_s, :self.k2_s*self.stride_s - self.step_col_s, :]
        # Applaying the normalizers back
        scaler_1 = self.scalers[0]
        scaler_2 = self.scalers[1]
        
        fake_img_A_reshaped = fake_img_A.reshape((fake_img_A.shape[0] * fake_img_A.shape[1], fake_img_A.shape[2]))
        fake_img_B_reshaped = fake_img_B.reshape((fake_img_B.shape[0] * fake_img_B.shape[1], fake_img_B.shape[2]))
        
        fake_img_inv_A = scaler_1.inverse_transform(fake_img_A_reshaped)
        fake_img_inv_B = scaler_2.inverse_transform(fake_img_B_reshaped)
        
        fake_img_norm_A = fake_img_inv_A.reshape((fake_img_A.shape[0], fake_img_A.shape[1], fake_img_A.shape[2]))
        fake_img_norm_B = fake_img_inv_B.reshape((fake_img_B.shape[0], fake_img_B.shape[1], fake_img_B.shape[2]))
        # Saving the fake images
        #saving generated images in .npy to use their in the classifier evaluation
        np.save(self.save_path + 'Adapted_Target', fake_img_norm_A)
        np.save(self.save_path + 'Adapted_Source', fake_img_norm_B)
        #saving generated images in .mat to use in visualization purposes
        sio.savemat(self.save_path + 'Adapted_Target.mat', {'fake_A': fake_img_norm_A})
        sio.savemat(self.save_path + 'Adapted_Source.mat', {'fake_B': fake_img_norm_B})
        if self.opt.save_real:
            real_img_A = self.patchcontainer_real_A[:self.k1_s*self.stride_s - self.step_row_s, :self.k2_s*self.stride_s - self.step_col_s, :]
            real_img_B = self.patchcontainer_real_B[:self.k1_t*self.stride_t - self.step_row_t, :self.k2_t*self.stride_t - self.step_col_t, :]
            real_img_A_reshaped = real_img_A.reshape((real_img_A.shape[0] * real_img_A.shape[1], real_img_A.shape[2]))
            real_img_B_reshaped = real_img_B.reshape((real_img_B.shape[0] * real_img_B.shape[1], real_img_B.shape[2]))
            real_img_inv_A = scaler_1.inverse_transform(real_img_A_reshaped)
            real_img_inv_B = scaler_2.inverse_transform(real_img_B_reshaped)
            
            real_img_norm_A = real_img_inv_A.reshape((real_img_A.shape[0], real_img_A.shape[1], real_img_A.shape[2]))
            real_img_norm_B = real_img_inv_B.reshape((real_img_B.shape[0], real_img_B.shape[1], real_img_B.shape[2]))
            #saving generated images in .mat to use in visualization purposes
            sio.savemat(self.save_path + 'Real_Source.mat', {'real_A': real_img_norm_A})
            sio.savemat(self.save_path + 'Real_Target.mat', {'real_B': real_img_norm_B})
        
        
        
        
        