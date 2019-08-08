import torch
import os, argparse
from srgan import SRGAN
from edsr import EDSR
from os.path import join
from os import listdir

"""parsing and configuration"""
def parse_args():
    desc = "PyTorch implementation of SR collections"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_name', type=str, default='SRGAN',
                        choices=['SRGAN','EDSR'], help='The type of model')
    parser.add_argument('--data_dir', type=str, default='../Data')
    parser.add_argument('--train_dataset', type=str, default='train',
                        help='The name of training dataset')
    parser.add_argument('--test_dataset', type=str, default='test',
                        help='The name of test dataset')
    parser.add_argument('--sample_set', nargs='+', default='all',help='select set of sample to train or test')
    parser.add_argument('--crop_size', type=int, default=128, help='Size of cropped HR image')
    parser.add_argument('--num_threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--num_channels', type=int, default=1, help='The number of channels to super-resolve')
    parser.add_argument('--num_residuals', type=int, default=8, help='The number of residual blocks')
    parser.add_argument('--scale_factor', type=int, default=2, choices=[2,4],help='Size of scale factor')
    parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--save_epochs', type=int, default=10, help='Save trained model every this epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--save_dir', type=str, default='Result', help='Directory name to save the results or directory to load model for test_whole_image mode')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gpu', dest='gpu_mode', action='store_true')
    parser.add_argument('--cpu', dest='gpu_mode', action='store_false')
    parser.set_defaults(gpu_mode=True)
    parser.add_argument('--registered', dest='registered', action='store_true')
    parser.add_argument('--downsample', dest='registered', action='store_false')
    parser.set_defaults(registered=True) 
    parser.add_argument('--mode',type=str,default='train_and_test', choices=['train_and_test','test','predict','predict3d'], help='train and test mode')
    parser.add_argument('--kernel', type=int, default=3, help='Size of kernel')
    parser.add_argument('--filter', type=int, default=128, help='Number of filters')
    parser.add_argument('--output', type=str, default='pic', choices=['pic', 'raw'],help='Output format for predict mode')
    parser.add_argument('--pretrain', type=int, default=0, help='Number of pretrain epoch for generator')
    parser.add_argument('--lr_d', type=float, default=0.1, help='Learning rate multiplier for discriminator')
    parser.add_argument('--vgg_factor', type=float, default=1.0, help='Factor for vgg loss')
    parser.add_argument('--vgg_layer', type=int, default=8, help='Layer for vgg loss')
    parser.add_argument('--metric', type=str, default='sc',
                        choices=['sc', 'ssim','psnr'], help='The type of metrics used for testing')
    parser.add_argument('--no_grayscale_correction', dest='grayscale_corrected', action='store_false')
    parser.set_defaults(grayscale_corrected=True)  

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --save_dir
    args.save_dir = os.path.join(args.save_dir, args.model_name) 
    if args.sample_set=='all':
        args.sample_set=listdir(join(args.data_dir,args.train_dataset))
    if args.mode is not 'predict':
        train_dataset=[]
        test_dataset=[]
        for sample in args.sample_set:
            train_dataset.append(join(args.train_dataset,sample))
            test_dataset.append(join(args.test_dataset,sample))
        args.train_dataset=train_dataset
        args.test_dataset=test_dataset
    print(args.train_dataset)
    print(args.test_dataset)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # --epoch
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    if args.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode=False")

    # model
    if args.model_name == 'SRGAN':
        net = SRGAN(args)
    elif args.model_name == 'EDSR':
        net = EDSR(args)
    else:
        raise Exception("[!] There is no option for " + args.model_name)
    
    if args.mode=='train_and_test':
        # train
        net.train()
        # test
        for test_dataset in args.test_dataset:    
            net.test(test_dataset)
    elif args.mode=='predict':
        for test_dataset in args.test_dataset:        
            net.predict2d(test_dataset,output=args.output)
    elif args.mode=='predict3d':
        for test_dataset in args.test_dataset:        
            net.predict3d(test_dataset,output='raw')        
    else:
        for test_dataset in args.test_dataset:  
            net.test(test_dataset)
    

if __name__ == '__main__':
    main()