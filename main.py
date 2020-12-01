### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os, argparse
import numpy as np
#from Model import MyModel
from Model import *
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="path to the data")
parser.add_argument("mode", help="train, test or predict")
#parser.add_argument("--save_dir", help="path to save the results")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel(training_configs)  
    
    #model = MyModel(training_configs)
    net = MyNetwork()
    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    start_epoch = 0
    best_acc = 0
    max_epoch = training_configs['max_epoch']

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        #print(checkpoint['net'])
        print(net.load_state_dict(checkpoint['net']))
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        #print(start_epoch)
        #print(start_epoch)
	
    
    
    
    
    if args.mode == 'train':
        
        x_train, y_train = load_data(args.data_dir)
        x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
        
        
        
        
        trainset = batch_data_process(x_train, x_valid, y_train, y_valid, training=True, validation=False, testing=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
        
        validset = batch_data_process(x_train, x_valid, y_train, y_valid, training=False, validation=True, testing=False)
        validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=0)
        
        for epoch in range(start_epoch, start_epoch+max_epoch):
            model.train(trainloader, epoch)
            model.evaluate(validloader, epoch)
            

    elif args.mode == 'test':
        
        x_train, y_train = load_data(args.data_dir)
        x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
        
        
        
        
        trainset = batch_data_process(x_train, x_valid, y_train, y_valid, training=True, validation=False, testing=False)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)
        
        validset = batch_data_process(x_train, x_valid, y_train, y_valid, training=False, validation=True, testing=False)
        validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=0)
        
        # Testing on public testing dataset
        #_, _, x_valid, y_valid = load_data(args.data_dir)
        
        print('\nEpoch: %d' % start_epoch)
        model.evaluate(validloader, start_epoch, net) 

    elif args.mode == 'predict':
        
        print('==>Predicting and storing results on private testing dataset') 
        
        x_test = load_testing_images(args.data_dir)
        testset = batch_data_process(None, x_test, None, None, training=False, validation=False, testing=True)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
        temp1 = np.zeros((0,10))
        
        for i in range(20):
            outputs = model.predict_prob(testloader, net)
            temp1 = np.concatenate((temp1,outputs),axis=0)
                  
            
        predictions = temp1
        
        
        

### END CODE HERE

