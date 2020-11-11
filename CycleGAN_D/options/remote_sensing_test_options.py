from .base_options import BaseOptions


class RemoteSensingTestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        # Images names for the evaluation
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
        parser.add_argument('--same_coordinates', dest='same_coordinates', type=bool, default=True, help='Decide if we use the same set of coordinates for both data')        
        parser.add_argument('--overlap_porcent_s', dest='overlap_porcent_s', type=float, default=.75, help='overlap porcent')
        parser.add_argument('--overlap_porcent_t', dest='overlap_porcent_t', type=float, default=.75, help='overlap porcent')
        parser.add_argument('--save_real', dest='save_real', type=bool, default= True, help='Decide if the real image is saved for verification' )
        self.isTrain = False
        return parser
