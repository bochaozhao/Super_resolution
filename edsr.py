#modified from https://github.com/togheppi/pytorch-super-resolution-model-collection/blob/master/edsr.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import base_network
from torch.utils.data import DataLoader
import network_utils
from torchvision.transforms import *
import dataset_utils
import utils
from os.path import join
import transforms_3d
from os import listdir
from PIL import Image
from torchvision import models
import numpy as np
import time

class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter=64, num_residuals=16,scale_factor=4,kernel=3):
        super(Net, self).__init__()
        pad=int((kernel-1)/2)
        #input size N*C*H*W
        self.input_conv = base_network.ConvBlock(num_channels, base_filter, kernel, 1, pad, activation=None, norm=None)
        #size N*F*H*W
        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(base_network.ResnetBlock(base_filter, kernel_size=kernel,padding=pad,norm=None))
        self.residual_layers = nn.Sequential(*resnet_blocks)
        #size N*F*H*W
        self.mid_conv = base_network.ConvBlock(base_filter, base_filter, kernel, 1, pad, activation=None, norm=None)
        
        upscale_blocks=[]
        
        
        for _ in range(scale_factor//2):
            upscale_blocks.append(base_network.Upsample2xBlock(base_filter, base_filter, upsample='ps', activation=None, norm=None))
        
        self.upscale4x = nn.Sequential(*upscale_blocks)
        #two pixel shuffle upsample block, final size N*F*4H*4W
        self.output_conv = base_network.ConvBlock(base_filter, num_channels, kernel, 1, pad, activation=None, norm=None)
        #size N*C*4H*4W
    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            network_utils.weights_init_normal(m, mean=mean, std=std)

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.output_conv(out)
        return out


class FeatureExtractor(torch.nn.Module):
    def __init__(self, netVGG, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(netVGG.features.children())[:(feature_layer + 1)])
        for param in self.features.parameters():
            param.requires_grad=False
    def forward(self, x):
        return self.features(x)    
    
    
class EDSR(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset #possible values: bsds500
        self.test_dataset = args.test_dataset   #possible values: bsds500
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
        self.filter=args.filter
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
        vgg_factor=self.vgg_factor
        train_dataset=[]
        # networks, number of filters and resiudal blocks
        self.model = Net(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=self.kernel)

        # weigh initialization
        self.model.weight_init()
        # For the content loss
        self.feature_extractor = FeatureExtractor(models.vgg19(pretrained=True),feature_layer=self.vgg_layer)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        # loss function
        if self.gpu_mode:
            print('in gpu mode')
            self.model.cuda()
            self.feature_extractor.cuda()
            self.L1_loss = nn.L1Loss().cuda()
        else:
            print('in cpu mode')
            self.L1_loss = nn.L1Loss()

        print('---------- Networks architecture -------------')
        network_utils.print_network(self.model)
        print('----------------------------------------------')

        # load dataset

        train_data_loader = self.load_ct_dataset(dataset=self.train_dataset, 
                                                     is_train=True,
                                                     is_registered=self.registered,
                                                     grayscale_corrected=self.grayscale_corrected)
        test_data_loader = self.load_ct_dataset(dataset=self.test_dataset, 
                                                    is_train=False,
                                                    is_registered=self.registered,
                                                    grayscale_corrected=self.grayscale_corrected)
        

        # set the logger
        #log_dir = os.path.join(self.save_dir, 'logs')
        #if not os.path.exists(log_dir):
        #    os.makedirs(log_dir)
        #logger = Logger(log_dir)

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        step = 0

        # test image
        test_lr, test_hr, test_bc = test_data_loader.dataset.__getitem__(2)
        test_lr = test_lr.unsqueeze(0)
        test_hr = test_hr.unsqueeze(0)
        test_bc = test_bc.unsqueeze(0)

        self.model.train()
        for epoch in range(self.num_epochs):
            if epoch==0:
                start_time=time.time()
            # learning rate is decayed by a factor of 2 every 40 epochs
            if (epoch+1) % 40 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2.0
                print('Learning rate decay: lr={}'.format(self.optimizer.param_groups[0]['lr']))

            epoch_loss = 0
            for iter, (lr, hr, _) in enumerate(train_data_loader):
                # input data (low resolution image)
                if self.num_channels == 1:
                    x_ = hr[:, 0].unsqueeze(1)
                    y_ = lr[:, 0].unsqueeze(1)
                else:
                    x_ = hr
                    y_ = lr

                if self.gpu_mode:
                    x_ = x_.cuda()
                    y_ = y_.cuda()

                # update network
                self.optimizer.zero_grad()
                recon_image = self.model(y_)
                
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
                vgg_loss=vgg_loss*vgg_factor
                
                loss = self.L1_loss(recon_image, x_)+vgg_loss
                loss.backward()
                self.optimizer.step()

                # log
                epoch_loss += loss.item()
                #print('Epoch: [%2d] [%4d/%4d] loss: %.8f' % ((epoch + 1), (iter + 1), len(train_data_loader), loss.item()))
                print('Epoch: [%2d] [%4d/%4d] loss: %.8f vggloss: %.8f' % ((epoch + 1), (iter + 1), len(train_data_loader), loss.item(),vgg_loss.item()))
                # tensorboard logging
                #logger.scalar_summary('loss', loss.data[0], step + 1)
                #step += 1

            # avg. loss per epoch
            avg_loss.append(epoch_loss / len(train_data_loader))

            # prediction
            if self.num_channels == 1:
                y_ = test_lr[:, 0].unsqueeze(1)
            else:
                y_ = test_lr

            if self.gpu_mode:
                y_ = y_.cuda()

            recon_img = self.model(y_)
            sr_img = recon_img[0].cpu()

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
        network_utils.plot_loss([avg_loss], self.num_epochs, save_dir=save_dir)
        print('Training is finished.')

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def test(self,test_dataset):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=self.kernel)

        if self.gpu_mode:
            self.model.cuda()

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
        self.model.eval()
        
        metric=[]
        for lr, hr, bc in test_data_loader:
            # input data (low resolution image)
            if self.num_channels == 1:
                y_ = lr[:, 0].unsqueeze(1)
            else:
                y_ = lr

            if self.gpu_mode:
                y_ = y_.cuda()

                # prediction
            recon_imgs = self.model(y_)
            #sr_sr_imgs = self.model(recon_imgs)
            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                sr_img = recon_img.cpu()
                #sr_sr_img=sr_sr_imgs[i].cpu()
                # save result image
      
                save_dir = os.path.join(self.save_dir, test_dataset)

                network_utils.save_img(sr_img, img_num, save_dir=save_dir)
                #network_utils.save_img(sr_sr_img, img_num,  os.path.join(save_dir, 'sr-sr'))

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
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=self.kernel)

        if self.gpu_mode:
            self.model.cuda()

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
            if self.registered:
                img = transforms_3d.rescale(img,original_scale=(minvalue,maxvalue),new_scale=(0,1))           
                img = Image.fromarray(img)
                lr_transform = Compose([ToTensor()])
                lr_img = lr_transform(img)   
            else:        
                img = transforms_3d.rescale(img,original_scale=(minvalue,maxvalue),new_scale=(0,1))
                img = Image.fromarray(img)
            
                # lr_img LR image
                lr_transform = Compose([ToTensor()])
                lr_img = lr_transform(img)

    

            if self.num_channels == 1:
                y_ = lr_img.unsqueeze(1)
            else:
                raise Exception(" test_single only accept 2d raw image file " )

            if self.gpu_mode:
                y_ = y_.cuda()

            # prediction
            self.model.eval()
            recon_img = self.model(y_)
            
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
        self.model = Net(num_channels=self.num_channels, base_filter=self.filter, num_residuals=self.num_residuals,scale_factor=self.scale_factor,kernel=self.kernel)

        if self.gpu_mode:
            self.model.cuda()

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
            self.model.eval()
            recon_img = self.model(y_)
            
            
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
            
            
            
            
    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if epoch is not None:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name +
                       '_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                       % (self.num_channels, self.batch_size, epoch, self.lr))
        else:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name +
                       '_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                       % (self.num_channels, self.batch_size, self.num_epochs, self.lr))

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')
        model_name = model_dir + '/' + self.model_name +\
                     '_param_ch%d_batch%d_epoch%d_lr%.g.pkl'\
                     % (self.num_channels, self.batch_size, self.num_epochs, self.lr)
        if os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False