import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import base_network
from torch.utils.data import DataLoader
import network_utils
import dataset_utils
import utils
from os.path import join
from os import listdir
from PIL import Image
from torchvision.transforms import *
import numpy as np
import transforms_3d
import time

'''Generator architecture'''
class Generator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals,scale_factor=4,kernel=3):
        super(Generator, self).__init__()
        pad=int((kernel-1)/2)
        #input size N*C*H*W
        self.input_conv = base_network.ConvBlock(num_channels, base_filter,kernel, 1, pad, activation=None, norm=None)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(base_network.ResnetBlock(base_filter, kernel_size=kernel, padding=pad,norm=None))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = base_network.ConvBlock(base_filter, base_filter, kernel, 1, pad, activation=None,norm=None)
        upscale_blocks=[]
        for _ in range(scale_factor//2):
            upscale_blocks.append(base_network.Upsample2xBlock(base_filter, base_filter, upsample='ps', activation=None, norm=None))
        
        self.upscale4x = nn.Sequential(*upscale_blocks)


        self.output_conv = base_network.ConvBlock(base_filter, num_channels, kernel, 1, pad, activation=None, norm=None)
        #size N*C*4H*4W
    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.output_conv(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            network_utils.weights_init_normal(m, mean=mean, std=std)

'''Discriminator architecture'''
class Discriminator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        #input size N*C*H*W
        self.input_conv = base_network.ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None)

        self.conv_blocks = nn.Sequential(
            base_network.ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu'),
            base_network.ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu'),
            base_network.ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu'),
            base_network.ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu'),
            base_network.ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu'),
            #base_network.ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu'),
            #base_network.ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu'),
        )

        self.dense_layers = nn.Sequential(
            base_network.DenseBlock(base_filter * 4 * image_size // 8 * image_size // 8, base_filter * 8, activation='lrelu',norm=None),
            base_network.DenseBlock(base_filter * 8, 1, activation='sigmoid', norm=None)
        )

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.dense_layers(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            network_utils.weights_init_normal(m, mean=mean, std=std)

'''Feature Extractor architecture using pretrained VGG19 net'''
class FeatureExtractor(torch.nn.Module):
    def __init__(self, netVGG, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(netVGG.features.children())[:(feature_layer + 1)])
        for param in self.features.parameters():
            param.requires_grad=False
    def forward(self, x):
        return self.features(x)

'''SRGAN architecture'''
class SRGAN(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.registered=args.registered
        self.mode=args.mode
        self.kernel=args.kernel
        self.epoch_pretrain=args.pretrain
        self.filter=args.filter
        self.lr_d=args.lr_d
        self.vgg_factor=args.vgg_factor
        self.vgg_layer=args.vgg_layer
        self.grayscale_corrected=args.grayscale_corrected
        self.metric=args.metric
        self.num_residuals=args.num_residuals
        self.sample_set=args.sample_set


        
    def load_ct_dataset(self, dataset, is_train=True,is_registered=False,grayscale_corrected=True):
        '''
        dataset: list of dataset such as ['train\GB1', 'train\B1M6'] 
        '''
        if is_train:
            print('Loading train ct datasets...')
            train_set = utils.get_training_set(self.data_dir, 
                                               dataset,
                                               self.crop_size,
                                               self.scale_factor,
                                               registered=is_registered,
                                               grayscale_corrected=grayscale_corrected)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        else:
            print('Loading test ct datasets...')
            test_set = utils.get_test_set(self.data_dir, 
                                              dataset,
                                              self.crop_size, 
                                              self.scale_factor,
                                              registered=is_registered,
                                              grayscale_corrected=grayscale_corrected)

            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)  
        

    
        
    def train(self):
        #defining weight factor for GAN loss, MSE loss and VGG loss for the loss function and label smoothing factor for discriminator
        gan_factor=0.1
        mse_factor=1
        vgg_factor=self.vgg_factor
        smooth_factor=0.1
        
        train_dataset=[]
        

        # load dataset

        train_data_loader = self.load_ct_dataset(dataset=self.train_dataset, is_train=True,
                                                     is_registered=self.registered,
                                                     grayscale_corrected=self.grayscale_corrected)
        test_data_loader = self.load_ct_dataset(dataset=self.test_dataset, is_train=False,
                                                    is_registered=self.registered,
                                                    grayscale_corrected=self.grayscale_corrected)            


        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=self.kernel)
        self.D = Discriminator(num_channels=self.num_channels, base_filter=self.filter, image_size=self.crop_size)

        # weigh initialization
        self.G.weight_init()
        self.D.weight_init()

        # For the content loss
        self.feature_extractor = FeatureExtractor(models.vgg19(pretrained=True),feature_layer=self.vgg_layer)

        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr*self.lr_d, betas=(0.9, 0.999))

        # loss function
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.feature_extractor.cuda()
            self.L1_loss = nn.L1Loss().cuda()
            self.MSE_loss = nn.MSELoss().cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()
            self.BCE_loss = nn.BCELoss()
            self.L1_loss = nn.L1Loss()

        print('---------- Networks architecture -------------')
        network_utils.print_network(self.G)
        network_utils.print_network(self.D)
        print('----------------------------------------------')


        ################# Pre-train generator #################

        # Load pre-trained parameters of generator
        if not self.load_model(is_pretrain=True):
            # Pre-training generator for 50 epochs
            print('Pre-training is started.')
            self.G.train()
            for epoch in range(self.epoch_pretrain):
                for iter, (lr, hr, _) in enumerate(train_data_loader):
                    # input data (low resolution image)
                    if self.num_channels == 1:
                        x_ = hr
                        y_ = lr
                        #x_ = network_utils.norm(hr.repeat(1,3,1,1), vgg=True)
                        #x_ = torch.mean(x_,1,True)
                        #y_ = network_utils.norm(lr.repeat(1,3,1,1), vgg=True)
                        #y_ = torch.mean(y_,1, True)
                    else:
                        x_ = network_utils.norm(hr, vgg=True)
                        y_ = network_utils.norm(lr, vgg=True)

                    if self.gpu_mode:
                        x_ = x_.cuda()
                        y_ = y_.cuda()

                    # Train generator
                    self.G_optimizer.zero_grad()
                    recon_image = self.G(y_)

                    # Content losses
                    content_loss = self.L1_loss(recon_image, x_)

                    # Back propagation
                    G_loss_pretrain = content_loss
                    G_loss_pretrain.backward()
                    self.G_optimizer.step()

                    # log
                    print("Epoch: [%2d] [%4d/%4d] G_loss_pretrain: %.8f"
                          % ((epoch + 1), (iter + 1), len(train_data_loader), G_loss_pretrain.item()))

            print('Pre-training is finished.')

            # Save pre-trained parameters of generator
            self.save_model(is_pretrain=True)

        ################# Adversarial train #################
        print('Training is started.')
        # Avg. losses
        G_avg_loss = []
        D_avg_loss = []
        step = 0

        # test image
        test_lr, test_hr, test_bc = test_data_loader.dataset.__getitem__(20)
        test_lr = test_lr.unsqueeze(0)
        test_hr = test_hr.unsqueeze(0)
        test_bc = test_bc.unsqueeze(0)

        self.G.train()
        self.D.train()
        for epoch in range(self.num_epochs):
            self.G.train()
            self.D.train()
            if epoch==0:
                start_time=time.time()
            # learning rate is decayed by a factor of 2 every 40 epoch
            if (epoch + 1) % 40 == 0:
                for param_group in self.G_optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay for G: lr={}".format(self.G_optimizer.param_groups[0]["lr"]))
                for param_group in self.D_optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay for D: lr={}".format(self.D_optimizer.param_groups[0]["lr"]))

            G_epoch_loss = 0
            D_epoch_loss = 0
            for iter, (lr, hr, _) in enumerate(train_data_loader):
                # input data (low resolution image)
                mini_batch = lr.size()[0]

                if self.num_channels == 1:
                    x_ = hr
                    y_ = lr
                    
                else:
                    x_ = network_utils.norm(hr, vgg=True)
                    y_ = network_utils.norm(lr, vgg=True)

                if self.gpu_mode:
                    x_ = x_.cuda()
                    y_ = y_.cuda()
                    # labels
                    real_label = torch.ones(mini_batch).cuda()
                    fake_label = torch.zeros(mini_batch).cuda()
                else:
                    # labels
                    real_label = torch.ones(mini_batch)
                    fake_label = torch.zeros(mini_batch)

                # Reset gradient
                self.D_optimizer.zero_grad()

                # Train discriminator with real data
                D_real_decision = self.D(x_)
                D_real_loss = self.BCE_loss(D_real_decision.squeeze(),real_label*(1.0-smooth_factor))

                # Train discriminator with fake data
                recon_image = self.G(y_)
                D_fake_decision = self.D(recon_image)
                D_fake_loss = self.BCE_loss(D_fake_decision.squeeze(), fake_label)
                
                D_loss = (D_real_loss + D_fake_loss)*gan_factor

                # Back propagation
                D_loss.backward()
                self.D_optimizer.step()

                # Reset gradient
                self.G_optimizer.zero_grad()

                # Train generator
                recon_image = self.G(y_)
                D_fake_decision = self.D(recon_image)

                # Adversarial loss
                GAN_loss = self.BCE_loss(D_fake_decision.squeeze(), real_label)

                # Content losses
                mse_loss = self.L1_loss(recon_image, x_)
                
                if self.num_channels == 1:
                    x_VGG=hr.repeat(1,3,1,1).cpu()
                    x_VGG = network_utils.norm(x_VGG, vgg=True)
                    recon_VGG=recon_image.repeat(1,3,1,1).cpu()
                    recon_VGG = network_utils.norm(recon_VGG, vgg=True)
                else:
                    x_VGG = network_utils.norm(hr.cpu(), vgg=True)
                    recon_VGG = network_utils.norm(recon_image.cpu(), vgg=True)
                if self.gpu_mode:
                    x_VGG=x_VGG.cuda()
                    recon_VGG=recon_VGG.cuda()
                real_feature = self.feature_extractor(x_VGG)
                fake_feature = self.feature_extractor(recon_VGG)
                vgg_loss = self.L1_loss(fake_feature, real_feature.detach())

                # Back propagation
                mse_loss=mse_factor*mse_loss
                vgg_loss=vgg_factor*vgg_loss
                GAN_loss=gan_factor*GAN_loss
                G_loss = mse_loss +  vgg_loss + GAN_loss
                G_loss.backward()
                self.G_optimizer.step()

                # log
                G_epoch_loss += G_loss.item()
                D_epoch_loss += D_loss.item()
                #print("Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f"
                #      % ((epoch + 1), (iter + 1), len(train_data_loader), G_loss.item(), D_loss.item()))

                print("Epoch: [%2d] [%4d/%4d] G_loss: %.8f, mse: %.4f,vgg: %.4f, gan: %.4f,D_loss: %.8f"
                      % ((epoch + 1), (iter + 1), len(train_data_loader), G_loss.item(), mse_loss.item(),vgg_loss.item(),GAN_loss.item(),D_loss.item()))
                

                step += 1

            # avg. loss per epoch
            G_avg_loss.append(G_epoch_loss / len(train_data_loader))
            D_avg_loss.append(D_epoch_loss / len(train_data_loader))

            # prediction
            if self.num_channels == 1:
                y_ = test_lr
                #y_ = network_utils.norm(test_lr.repeat(1,3,1,1), vgg=True)
                #y_ = torch.mean(y_,1,True)
            else:
                y_ = network_utils.norm(test_lr, vgg=True)

            if self.gpu_mode:
                y_ = y_.cuda()

            recon_img = self.G(y_)
            if self.num_channels == 1:
                sr_img=recon_img.cpu()
                #sr_img=network_utils.denorm(recon_img.repeat(1,3,1,1).cpu(),vgg=True)
                #sr_img=torch.mean(sr_img,1,True)
            else:
                sr_img = network_utils.denorm(recon_img.cpu(), vgg=True)

            sr_img=sr_img[0]
            # save result image
            save_dir = os.path.join(self.save_dir, 'train_result')
            network_utils.save_img(sr_img, epoch + 1, save_dir=save_dir, is_training=True)
            if epoch==0:
                print('time for 1 epoch is :%.2f'%(time.time()-start_time))
            print('Result image at epoch %d is saved.' % (epoch + 1))

            # Save trained parameters of model
            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(epoch + 1)



        # calculate psnrs
        if self.num_channels == 1:
            gt_img = test_hr[0][0].unsqueeze(0)
            lr_img = test_lr[0][0].unsqueeze(0)
            bc_img = test_bc[0][0].unsqueeze(0)
        else:
            gt_img = test_hr[0]
            lr_img = test_lr[0]
            bc_img = test_bc[0]
        if self.metric=='sc':
            bc_metric = network_utils.SC(bc_img, gt_img)
            recon_metric = network_utils.SC(sr_img, gt_img)
        elif self.metric=='ssim':
            bc_metric = network_utils.SSIM(bc_img, gt_img)
            recon_metric = network_utils.SSIM(sr_img, gt_img)
        else:
            bc_metric = network_utils.PSNR(bc_img, gt_img)
            recon_metric = network_utils.PSNR(sr_img, gt_img)

        # plot result images
        result_imgs = [gt_img, lr_img, bc_img, sr_img]
        metrics = [None, None, bc_metric, recon_metric]
        network_utils.plot_test_result(result_imgs, metrics, self.num_epochs, save_dir=save_dir, is_training=True, index=self.metric)
        print('Training result image is saved.')


        # Plot avg. loss
        network_utils.plot_loss([G_avg_loss, D_avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def test(self,test_dataset):
        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=3)

        if self.gpu_mode:
            self.G.cuda()

        # load model
        self.load_model()

        
        # load dataset
        test_data_loader = self.load_ct_dataset(dataset=[test_dataset], 
                                                    is_train=False,
                                                    is_registered=self.registered,
                                                    grayscale_corrected=self.grayscale_corrected)      


        # Test
        print('Test is started.')
        img_num = 0
        total_img_num = len(test_data_loader)
        self.G.eval()
        metric=[]
        for lr, hr, bc in test_data_loader:

            # input data (low resolution image)
            if self.num_channels == 1:
                y_ = lr[:, 0].unsqueeze(1)
            else:
                y_ = network_utils.norm(lr, vgg=True)

            if self.gpu_mode:
                y_ = y_.cuda()

            # prediction
            recon_imgs = self.G(y_)
            
            if self.num_channels == 1:
                recon_imgs=recon_imgs.cpu()
            else:
                recon_imgs = network_utils.denorm(recon_imgs.cpu(), vgg=True)

            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                sr_img = recon_img

                # save result image
                save_dir = os.path.join(self.save_dir, test_dataset)
                network_utils.save_img(sr_img, img_num, save_dir=save_dir)

                # calculate psnrs
                if self.num_channels == 1:
                    gt_img = hr[i][0].unsqueeze(0)
                    lr_img = lr[i][0].unsqueeze(0)
                    bc_img = bc[i][0].unsqueeze(0)
                else:
                    gt_img = hr[i]
                    lr_img = lr[i]
                    bc_img = bc[i]

                if self.metric=='sc':
                    bc_metric = network_utils.SC(bc_img, gt_img)
                    recon_metric = network_utils.SC(sr_img, gt_img)
                elif self.metric=='ssim':
                    bc_metric = network_utils.SSIM(bc_img, gt_img)
                    recon_metric = network_utils.SSIM(sr_img, gt_img)
                else:
                    bc_metric = network_utils.PSNR(bc_img, gt_img)
                    recon_metric = network_utils.PSNR(sr_img, gt_img)
                    
                metric.append(recon_metric)
                # plot result images
                result_imgs = [gt_img, lr_img, bc_img, sr_img]
                metrics = [None, None, bc_metric, recon_metric]
                network_utils.plot_test_result(result_imgs, metrics, img_num, save_dir=save_dir, index=self.metric)


                print('Test DB: %s, Saving result images...[%d/%d]' % (test_dataset, img_num, total_img_num))

        print('Test is finishied.')
        mean_metric=np.mean(metric)
        std_metric=np.std(metric)
        save_fn = save_dir + '\\results.txt'
        with open(save_fn,'w+') as file:
            file.write('average metric value is: %.3f\n' %mean_metric)
            file.write('std of metric value is: %.3f' %std_metric)

    def predict2d(self,test_dataset,output='pic'):
        '''take 2d image and perform super resolution'''
        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=3)

        if self.gpu_mode:
            print('gpu mode')
            self.G.cuda()

        # load model
        self.load_model()
            
        image_dir=join(self.data_dir, test_dataset)
                         
        image_filenames=[]  
        image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if utils.is_raw_file(x))
                
            
        img_num=0
        for img_fn in image_filenames:
            print(img_fn)
          
            img = utils.read_and_reshape(img_fn).astype(float)
            minvalue=img.min()
            maxvalue=img.max()
            img = transforms_3d.rescale(img,original_scale=(minvalue,maxvalue),new_scale=(0,1))        
            img = Image.fromarray(img)
            lr_transform = Compose([ToTensor()])
            lr_img = lr_transform(img)   

            if self.num_channels == 1:
                y_ = lr_img.unsqueeze(0)
            else:
                raise Exception("only accept 2d raw image file " )

            if self.gpu_mode:
                y_ = y_.cuda()
            
            # prediction
            self.G.eval()
            recon_img = self.G(y_)
            

            recon_img = recon_img.cpu()[0].clamp(0, 1).detach().numpy()
            if output=='raw':
                recon_img = transforms_3d.rescale(recon_img,original_scale=(0,1),new_scale=(minvalue,maxvalue)).astype(int)  
                img_filename=img_fn.split('\\')[-1]            
                save_dir = os.path.join(self.save_dir, 'SR-2D-raw',test_dataset)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                utils.save_as_raw(recon_img,os.path.join(save_dir,img_filename),dtype='uint16',prefix='SR')
            else:
                recon_img=torch.from_numpy(recon_img)    
                save_dir = os.path.join(self.save_dir, 'SR-2D-png', test_dataset)

                network_utils.save_img(recon_img, img_num, save_dir=save_dir)
            img_num+=1
            torch.cuda.empty_cache() 
        print('Single test result image is saved.')

        
        
        
    def predict3d(self,test_dataset,output='raw'):
        '''take 3D image and do x, y and z directional slice. Then perform super resolution on each group. This step is followed by combining the x,y and z directional results together into a new 3D image. This method only supports input of raw file'''
        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=3)

        if self.gpu_mode:
            print('gpu mode')
            self.G.cuda()

        # load model
        self.load_model()
            
        image_dir=join(self.data_dir, test_dataset)
        print(image_dir)
        utils.resample_and_save2d(image_dir,axis='x',dtype='uint16')
        utils.resample_and_save2d(image_dir,axis='y',dtype='uint16')
        utils.resample_and_save2d(image_dir,axis='z',dtype='uint16')
        
        image_filenames=[]  
        image_filenames.extend(join(image_dir, 'x_slice', x) for x in sorted(listdir(join(image_dir,'x_slice'))) if utils.is_raw_file(x))
        image_filenames.extend(join(image_dir, 'y_slice', x) for x in sorted(listdir(join(image_dir,'y_slice'))) if utils.is_raw_file(x))        
        image_filenames.extend(join(image_dir,'z_slice',  x) for x in sorted(listdir(join(image_dir,'z_slice'))) if utils.is_raw_file(x))
                               
        img_num=0
        for img_fn in image_filenames:
            print(img_fn)
          
            img = utils.read_and_reshape(img_fn).astype(float)

            minvalue=img.min()
            maxvalue=img.max() 
            img = transforms_3d.rescale(img,original_scale=(minvalue,maxvalue),new_scale=(0,1))        
            img = Image.fromarray(img)
            lr_transform = Compose([ToTensor()])
            lr_img = lr_transform(img)   



            if self.num_channels == 1:
                y_ = lr_img.unsqueeze(0)
            else:
                raise Exception("only accept 3d raw image file " )

            if self.gpu_mode:
                y_ = y_.cuda()
            
            # prediction
            self.G.eval()
            recon_img = self.G(y_)           
            recon_img = recon_img.cpu()[0].clamp(0, 1).detach().numpy()
            recon_img = transforms_3d.rescale(recon_img,original_scale=(0,1),new_scale=(minvalue,maxvalue)).astype(int)  
            img_filename=img_fn.split('\\')[-2:]
            save_dir = os.path.join(self.save_dir, 'SR-3D',test_dataset)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            x_dir=os.path.join(save_dir,'x_slice')  
            if not os.path.exists(x_dir):
                os.makedirs(x_dir)
                    
            y_dir=os.path.join(save_dir,'y_slice')  
            if not os.path.exists(y_dir):
                os.makedirs(y_dir)                    
                    
            z_dir=os.path.join(save_dir,'z_slice')  
            if not os.path.exists(z_dir):
                os.makedirs(z_dir)                    
                    
                    
            utils.save_as_raw(recon_img,os.path.join(save_dir,img_filename[0],img_filename[1]),dtype='uint16',prefix='SR')

            img_num+=1
            torch.cuda.empty_cache()    
        print(save_dir)    
        transforms_3d.combine_2d_slices(save_dir,scale_factor=self.scale_factor)    
        print('Predicted images are saved.')        
        
        
        
    def save_model(self, epoch=None, is_pretrain=False):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if is_pretrain:
            torch.save(self.G.state_dict(), model_dir + '/' + self.model_name + '_G_param_pretrain.pkl')
            print('Pre-trained generator model is saved.')
        else:
            if epoch is not None:
                torch.save(self.G.state_dict(), model_dir + '/' + self.model_name +
                           '_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))
                torch.save(self.D.state_dict(), model_dir + '/' + self.model_name +
                           '_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))
            else:
                torch.save(self.G.state_dict(), model_dir + '/' + self.model_name +
                           '_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
                torch.save(self.D.state_dict(), model_dir + '/' + self.model_name +
                           '_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
            print('Trained models are saved.')

    def load_model(self, is_pretrain=False):
        model_dir = os.path.join(self.save_dir, 'model')

        if is_pretrain:
            model_name = model_dir + '/' + self.model_name + '_G_param_pretrain.pkl'
            if os.path.exists(model_name):
                self.G.load_state_dict(torch.load(model_name))
                print('Pre-trained generator model is loaded.')
                return True
        else:
            model_name = model_dir + '/' + self.model_name + \
                         '_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl' \
                         % (self.num_channels, self.batch_size, self.num_epochs, self.lr)
            if os.path.exists(model_name):
                self.G.load_state_dict(torch.load(model_name))
                print('Trained generator model is loaded.')
                return True

        return False

