import torch
from .base_model import BaseModel
from . import networks


def getImageSpectrum( im, ny, nx ):
    """Calculates log-scaled magnitude of Fourier spectrum.
        Returns image scaled from -1 to +1 range"""

    im_fft = torch.rfft(im,2,normalized=True,onesided=False)
        
    im_fft = torch.view_as_complex(im_fft)

    im_fft = im_fft.real**2+im_fft.imag**2

    im_fft = torch.roll(im_fft,ny//2,2)
    im_fft = torch.roll(im_fft,nx//2,3)

    im_fft = torch.log(im_fft+0.001) # small offset to provent log(0)=NaN

    return im_fft


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--target_real_label', type=float, default=1.0,
                    help='Discriminator real target. Set to <1.0 for one-sided label smoothing')
            parser.add_argument('--lambda_gp', type=float, default=0.0, help='gradient penalty weighting, for wgangp')
            parser.add_argument('--instance_noise', type=float, default=0.0, help='add noise to discriminator inputs to stabilize training')

        parser.add_argument('--lambda_FFT', type=float, default=0.0, help='weight for Fourier spectrum matching L1 loss')
    

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.netD == 'none':
            self.loss_names = ['G_L1']
        else:
            self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        if opt.lambda_FFT > 0:
            self.loss_names.append('G_FFT')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if opt.lambda_FFT > 0:
            self.visual_names += ['real_B_FFT', 'fake_B_FFT']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and not opt.netD == 'none':
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain and not (opt.netD == 'none'):  
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if (self.opt.gan_mode == 'wgangp') and (self.opt.norm=='batch'):
                # disable normalization in a wasserstein discriminator (critic)
                opt.normD = 'none'
            else:
                opt.normD = opt.norm

            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if not (opt.netD == 'none'):

                self.criterionGAN = networks.GANLoss(opt.gan_mode, target_real_label=opt.target_real_label).to(self.device)
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        if self.opt.instance_noise > 0:
            fake_AB += torch.randn(fake_AB.size()).to(self.device)*self.opt.instance_noise

        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        if self.opt.instance_noise > 0:
            real_AB += torch.randn(fake_AB.size()).to(self.device)*self.opt.instance_noise

        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # add gradient penalty, in case of wasserstein loss being used
        if self.opt.gan_mode == 'wgangp':
            self.loss_D += networks.cal_gradient_penalty(self.netD, real_AB, fake_AB, self.device, lambda_gp=self.opt.lambda_gp)
            self.loss_D.backward(retain_graph=True)
        else:
            # calculate gradients
            self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if not (self.opt.netD == 'none'):
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            if self.opt.instance_noise > 0:
                fake_AB += torch.randn(fake_AB.size()).to(self.device)*self.opt.instance_noise
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # calculate Fourier domain loss if requested
        if self.opt.lambda_FFT > 0:
            self.real_B_FFT = getImageSpectrum( self.real_B, nx=self.opt.crop_size, ny=self.opt.crop_size )
            self.fake_B_FFT = getImageSpectrum( self.fake_B, nx=self.opt.crop_size, ny=self.opt.crop_size )
            self.loss_G_FFT = self.criterionL1( self.fake_B_FFT, self.real_B_FFT ) * self.opt.lambda_FFT

            self.loss_G += self.loss_G_FFT

        self.loss_G.backward()

    def optimize_parameters(self, enableGeneratorUpdate=True):
        self.forward()                   # compute fake images: G(A)
        # update D
        if not (self.opt.netD == 'none'):
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

        # update G
        if enableGeneratorUpdate:
            if not (self.opt.netD == 'none'):
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.backward_G()                   # calculate graidents for G
            self.optimizer_G.step()             # udpate G's weights
