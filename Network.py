import torch
import torch.nn as nn
import torch.nn.functional as F

############################################ begin of Architecture1 ##########################################################
class Myblock(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Myblock, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        #print(x.shape)
        y1 = self.b1(x)
        #print(y1.shape)
        y2 = self.b2(x)
        #print(y2.shape)
        y3 = self.b3(x)
        #print(y3.shape)
        y4 = self.b4(x)
        z = torch.cat([y1,y2,y3,y4], 1)
        #print(z.shape)
        #y5 = x
        #return torch.cat([y1,y2,y3,y4], 1)
        return z


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.a3 = Myblock(256,  64,  96, 128, 16, 32, 32)
        self.b3 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.b4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.c4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.d4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.e4 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.a5 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.b5 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        #print(x.shape)
        out = self.pre_layers(x)
        #print(out.shape)
        out = self.a3(out) 
        #print('rey')
        out = self.b3(out)
        #print(out.shape)
        out = self.maxpool(out)
        #print(out.shape)
        out1 = self.a4(out)
        #print(out1.shape)
        out = self.b4(out1)
        #print(out.shape)
        out = self.c4(out) 
        #print(out.shape)
        out = self.d4(out)
        #print(out.shape)
        out = self.e4(out)
        #print(out.shape)
        out = self.maxpool(out)
        #print(out.shape)
        out = self.a5(out)
        #print(out.shape)
        out = self.b5(out)
        #print(out.shape)
        out = self.avgpool(out)
        #print(out.shape)
        #print("rey avg")
        out = out.view(out.size(0), -1)
        #print(out.shape)
        out = self.linear(out)
        #print(out.shape)
        return out
  ###########################################end of architecture1#############################################################


######################################## begin of Architecture2 ####################################################

 '''
 class Myblock(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Myblock, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        
        y1 = self.b1(x)
        
        y2 = self.b2(x)
        
        y3 = self.b3(x)
        
        y4 = self.b4(x)
        z = torch.cat([y1,y2,y3,y4], 1)
        
        return z


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.a3 = Myblock(256,  64,  96, 128, 16, 32, 32)
        self.b3 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_1 = nn.Conv2d(256, 256, kernel_size=1, stride=2)
        self.conv2d_2 = nn.Conv2d(256, 256, kernel_size=1, stride=4)

        self.a4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.b4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.c4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.d4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.e4 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.a5 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.b5 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        
        out = self.pre_layers(x)
        #print(out.shape)
        
        out1 = self.a3(out) 
        
        out2 = self.b3(out1) 
        
        outm = self.maxpool(out2) 
        
        out3 = self.a4(outm) 
        
        #out4 = self.b4(out3) 
        
        #out5 = self.c4(out4) 
        
        out6 = self.d4(out3) 
        
        out7 = self.e4(out6) 
        
        outm2 = self.maxpool(out7)
        
        out8 = self.a5(outm2)  
        
        out9 = self.b5(out8) + self.conv2d_2(out)
        
        out10 = self.avgpool(out9) 
        
        
        out11 = out10.view(out10.size(0), -1)
        
        out12 = self.linear(out11)
        
        return out12
 '''
 ###############################end of architecture2############################################
 
 ##################################begin of architecture 3##########################
 '''
 class Myblock(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Myblock, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        
        y1 = self.b1(x)
        
        y2 = self.b2(x)
        
        y3 = self.b3(x)
        
        y4 = self.b4(x)
        z = torch.cat([y1,y2,y3,y4], 1)
        
        return z


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.a3 = Myblock(256,  64,  96, 128, 16, 32, 32)
        self.b3 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2d_1 = nn.Conv2d(256, 256, kernel_size=1, stride=2)
        self.conv2d_2 = nn.Conv2d(256, 256, kernel_size=1, stride=4)

        self.a4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.b4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.c4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.d4 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.e4 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.a5 = Myblock(256, 64,  96, 128, 16, 32, 32)
        self.b5 = Myblock(256, 64,  96, 128, 16, 32, 32)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(256, 10)

    def forward(self, x):
        
        out = self.pre_layers(x)
        
        out1 = self.a3(out) 
        
        out2 = self.b3(out1) 
        
        out = self.maxpool(out2) 
        
        out3 = self.a4(out) 
        
        out4 = self.b4(out3) 
        
        out5 = self.c4(out4) 
        
        out6 = self.d4(out5) 
        
        out7 = self.e4(out6) 
        
        out = self.maxpool(out7)
        
        out8 = self.a5(out)  
        
        out9 = self.b5(out8) + self.conv2d_2(out1) + self.conv2d_2(out2) + self.conv2d_1(out3) + self.conv2d_1(out4) + self.conv2d_1(out5) + self.conv2d_1(out6) + self.conv2d_1(out7)
        
        out10 = self.avgpool(out9) 
        
        
        out11 = out10.view(out10.size(0), -1)
        
        out12 = self.linear(out11)
        
        return out12

 '''