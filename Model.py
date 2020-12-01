### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record, batch_data_process
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from ImageUtils import progress_bar


"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        
        self.learning_rate = configs['learning_rate']
        #self.max_epoch = configs['max_epoch']
        self.batch_size = configs['batch_size']
        self.momentum = configs['momentum']
        self.weight_decay = configs['weight_decay']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = MyNetwork()
        self.net.to(device)
        if device == 'cuda':
           self.net = torch.nn.DataParallel(self.net)
           cudnn.benchmark = True
            
        

    def model_setup(self):
        
        
        # self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        pass
        

    

    def train(self, trainloader, epoch):
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        #print(scheduler)
        
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scheduler.step()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long()
            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
    
   
    def evaluate(self, validloader, epoch, net):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        best_acc = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for batch_idx, (inputs, targets) in enumerate(validloader):
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.long()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(validloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc            
            


    def predict_prob(self, testloader, net):
        self.net.eval()
        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            for batch_idx, (inputs) in enumerate(testloader):
                outputs = net(inputs)
       
        predicted_values = outputs
        softmax = nn.Softmax()
        predictions = softmax(predicted_values)
        device = 'cpu'
        predictions = predictions.to(device)
        predictions = predictions.numpy()
        np.save('./results_dir/predictions.npy', predictions)
        
        return predictions
        
    


### END CODE HERE