import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training, validation, testing):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    image = record.reshape((3,32,32))
    
    image = np.transpose(image, [1, 2, 0])
    
    ### END CODE HERE
    

    image = preprocess_image(image, training, validation, testing) # If any.
    
    image = np.transpose(image, [2, 0, 1])
    
    
    return image


def preprocess_image(image, training, validation, testing):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    
    if training:
        
        image= np.pad(image,((4,),(4,),(0,)),"constant")
        
        i=np.random.randint(8)
        j=np.random.randint(8)
        image=image[i:i+32,j:j+32,:]
        a=np.random.randint(0,2)
        if a==1:
            image=np.flip(image,axis=1)
	
    ### END CODE HERE	

	### YOUR CODE HERE
    image=(image-np.mean(image))/(np.std(image))
    ### END CODE HERE

    return image
    


# Other functions
### YOUR CODE HERE

def batch_data_process(x_train, x_test, y_train, y_test, training, validation, testing):
    """Use Parse record to the entire data set and perform data preprocessing.

    Args:
        x_train: An array of shape [50000,3072]. 
        y_train: An array of shape [50000,]
        x_test: An array of shape [10000, 3072].
        y_test: An array of shape [10000, ].
        training: A boolean. Determine whether it is in training mode.
        validation: A boolean. Determine whether it is in testing mode

    Returns:
        for training: Returns a list of number of samples, where each list item contains x_train as 3,32,32 and y_train as a scalar
        similar for validation and testing
    """
    ### YOUR CODE HERE
    
    output = []
    
    if training:
        for i in range(x_train.shape[0]):
            temp1 = parse_record(x_train[i], training, validation, testing)  # of shape 32,32,3
            output.append([temp1,y_train[i]])
            #print(output[0][0].shape)
            
    elif validation:
        for i in range(x_test.shape[0]):
            temp1 = parse_record(x_test[i], training, validation, testing)
            output.append([temp1, y_test[i]])
    elif testing:
        for i in range(x_test.shape[0]):
            temp1 = parse_record(x_test[i], training, validation, testing)
            output.append(temp1)
            
    
    return output



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



term_width = 0

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    #print(term_width)
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
### END CODE HERE