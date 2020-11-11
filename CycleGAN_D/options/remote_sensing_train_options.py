from .base_options import BaseOptions


class RemoteSensingTrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # Remote sensing Images parameters
        parser.add_argument('--compute_ndvi', dest='compute_ndvi', type=eval, choices=[True, False], default=True, help='Cumpute and stack the ndvi index to the rest of bands')
        parser.add_argument('--buffer', dest='buffer', type=eval, choices=[True, False], default=True, help='Decide wether a buffer around deforestated regions will be performed')
        parser.add_argument('--source_domain', required=True, help='Domain name for the source')
        parser.add_argument('--target_domain', required=True, help='Domain name for the source')
        parser.add_argument('--images_section', dest='images_section', type=str, default='Organized/Images/', help='Folder for the images')
        parser.add_argument('--reference_section', dest='reference_section', type=str, default='Organized/References/', help='Folder for the reference')
        parser.add_argument('--source_image_name_T1', required=True, help='Source Image name for the time 1')
        parser.add_argument('--source_image_name_T2', required=True, help='Source Image name for the time 2')
        parser.add_argument('--target_image_name_T1', required=True, help='Target Image name for the time 2')
        parser.add_argument('--target_image_name_T2', required=True, help='Target Image name for the time 1')
        parser.add_argument('--standard_normalization', dest='standard_normalization', type= bool, default=True, help='Decide if performs standard normalization')
        parser.add_argument('--same_coordinates', dest='same_coordinates', type=bool, default=False, help='Decide if we use the same set of coordinates for both data')        
        parser.add_argument('--stride_s', dest='stride_s', type=int, default=50, help='step for the extraction of the patches')
        parser.add_argument('--stride_t', dest='stride_t', type=int, default=50, help='step for the extraction of the patches')
        self.isTrain = True
        return parser
