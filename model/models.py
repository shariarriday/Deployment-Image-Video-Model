# PyTorch
import torch
from torchvision import models
from torch import cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
from selfonn import SelfONNLayer
import timm

def reset_function_generic(m):
    if hasattr(m,'reset_parameters') or hasattr(m,'reset_parameters_like_torch'): 
        # print(m) 
        if isinstance(m, SelfONNLayer):
            m.reset_parameters_like_torch() 
        else:
            m.reset_parameters()

class SqueezeLayer(nn.Module):
    
    def forward(self,x):
        x = x.squeeze(2)
        x = x.squeeze(2)
        return x 

class UnSqueezeLayer(nn.Module):
    
    def forward(self,x):
        x = x.unsqueeze(2).unsqueeze(3)
        return x 



class CNN_Classifier(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, class_num),
            nn.LogSoftmax(dim=1)
        )

        # Initialize weights for the linear layers only
        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.classifier[3].weight)
        self.classifier[3].bias.data.fill_(0.01)


    def forward(self,x):
        x = self.classifier(x)
        return x




class Self_B_ResBlock(nn.Module):
    def __init__(self, in_channels=3, channel1=8, channel2=16, channel3=8, resConnection=False,q_order=3):
        super().__init__()
        self.layer1 = SelfONNLayer(in_channels=in_channels,out_channels=channel1,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.Batch1 = nn.BatchNorm2d(channel1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        self.resConnection = resConnection
        if self.resConnection:
            self.layer2 = SelfONNLayer(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
            self.Batch2 = nn.BatchNorm2d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        else:
            self.layer2 = SelfONNLayer(in_channels=channel1,out_channels=channel2,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
            self.Batch2 = nn.BatchNorm2d(channel2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.layer3 = SelfONNLayer(in_channels=channel2,out_channels=channel3,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.Batch3 = nn.BatchNorm2d(channel3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()

    def forward(self,x):
        input = x.clone()
        x = self.tanh(self.Batch1(self.layer1(x)))
        x = self.tanh(self.Batch2(self.layer2(x)))
        x = self.tanh(self.Batch3(self.layer3(x)))
        if self.resConnection:
            x= self.tanh(x.clone()+ input)
            
            # x = torch.cat((x,input), 1)
        return x

class Self_MobileNet(nn.Module):
    def __init__(self, input_channel = 3, last_layer_channel = 32, class_num= 10):
        super().__init__()
        self.class_num = class_num
        self.selfONN = SelfONNLayer(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=3,mode='fast')
        self.batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=16, channel2=48, channel3=32, resConnection=True)
        self.NonResidual1 = Self_B_ResBlock(in_channels=32, channel1=32, channel2=48, channel3=16, resConnection=False)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=16, channel1=24, channel2=48, channel3=16, resConnection=True)
        self.NonResidual2 = Self_B_ResBlock(in_channels=16, channel1=24, channel2=32, channel3=48, resConnection=False)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=48, channel1=16, channel2=56, channel3=48, resConnection=True)
        self.NonResidual3 = Self_B_ResBlock(in_channels=48, channel1=8, channel2=48, channel3=36, resConnection=False)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=36, channel1=16, channel2=32, channel3=36, resConnection=True)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=36, channel1=8, channel2=32, channel3=36, resConnection=True)
        self.NonResidual4 = Self_B_ResBlock(in_channels=36, channel1=8, channel2=32, channel3=last_layer_channel, resConnection=False)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((7,7))
        self.Flatten = nn.Flatten()
        self.Dropout2d = nn.Dropout(p=0.1)
        self.self_MLP = CNN_Classifier(in_channels = int(49*last_layer_channel),class_num = self.class_num)
        
    def forward(self, x):
        x= self.tanh(self.batchnorm(self.selfONN(x)))
        x= self.BottleneckResidual1(x)
        x= self.NonResidual1(x)
        x= self.BottleneckResidual2(x)
        x= self.NonResidual2(x) 
        x= self.BottleneckResidual3(x)
        x= self.NonResidual3(x)
        x= self.BottleneckResidual4(x)
        x= self.BottleneckResidual5(x)
        x= self.NonResidual4(x)
        x= self.AdaptiveAvgPool2d(x)
        x = self.Flatten(x)
        x = self.Dropout2d(x)
        x= self.self_MLP(x)
        return x



class Self_DenseMobileNet(nn.Module):
    def __init__(self, input_channel = 3, last_layer_channel = 32, class_num= 10, q_order=3):
        super().__init__()
        self.class_num = class_num
        self.InputMLP = 10
        self.selfONN = SelfONNLayer(in_channels=input_channel,out_channels=32,kernel_size=3,stride=2,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast')
        self.batchnorm = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.tanh = nn.Tanh()
        self.BottleneckResidual1 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual2 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True,q_order=q_order)
        self.maxpool = nn.MaxPool2d(2)
        self.BottleneckResidual3 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual4 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual5 = Self_B_ResBlock(in_channels=32, channel1=96, channel2=96, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual6 = Self_B_ResBlock(in_channels=32, channel1=84, channel2=84, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual7 = Self_B_ResBlock(in_channels=32, channel1=72, channel2=72, channel3=32, resConnection=True,q_order=q_order)
        self.BottleneckResidual8 = Self_B_ResBlock(in_channels=32, channel1=64, channel2=64, channel3=32, resConnection=True,q_order=q_order)
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(1)
        self.Flatten = nn.Flatten()
        self.Dropout = nn.Dropout(p=0.2)
        self.self_MLP = CNN_Classifier(in_channels =5*32,class_num = self.class_num)
    
    def forward(self,x):
        x = self.tanh(self.batchnorm(self.selfONN(x)))
        x = self.BottleneckResidual1(x)
        x = self.BottleneckResidual2(x)
        ##############
        inMLP1 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual3(self.maxpool(x))
        x = self.BottleneckResidual4(x)
        ##############
        inMLP2 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual5(self.maxpool(x))
        x = self.BottleneckResidual6(x)
        ##############
        inMLP3 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP4 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        x = self.BottleneckResidual7(self.maxpool(x))
        x = self.BottleneckResidual8(x)
        ##############
        inMLP5 = self.Flatten(self.AdaptiveAvgPool2d(x)) # input for MLP

        inMLP = torch.cat((inMLP1, inMLP2, inMLP3, inMLP4, inMLP5), dim=1)
        
        
        x= self.Flatten(inMLP)
        
        x = self.Dropout(x)
        x= self.self_MLP(x)
        return x




def cnn_V1(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        # 10th layer (MLP)
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        # 11th layer (MLP)  
        torch.nn.Linear(in_features=4096, out_features=class_num, bias=True), 
        torch.nn.LogSoftmax(dim=1) 
    )  
    #
    return model 

def cnn_V2(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv) 
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=256, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.2, inplace=False),
        # 10th layer (MLP)   
        torch.nn.Linear(in_features=256, out_features=class_num, bias=True), 
        torch.nn.LogSoftmax(dim=1) 
    )
    #
    return model 

def cnn_V3(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 2nd layer (conv)
        torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 3rd layer (conv)
        torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
        torch.nn.ReLU(inplace=True),
        # 4th layer (conv)
        torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 5th layer (conv)
        torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 6th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # 7th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        # 8th layer (conv)
        torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 9th layer (MLP)
        torch.nn.Linear(in_features=25088, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    )
    #
    return model 


def cnn_V4(input_ch, class_num): 
    model = torch.nn.Sequential(
        # 1st layer (conv)
        torch.nn.Conv2d(input_ch, 20, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # Average pooling 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(), 
        # 2nd layer (MLP)
        # conv_output = 7*7*20= 980 
        torch.nn.Linear(in_features=980, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    )
    #
    return model 


class cnn_V5(nn.Module):
    
    def __init__(self, input_ch, class_num): 
        super(cnn_V5, self).__init__() 

        # 1st layer (conv)
        self.conv1 = cnn_V5.conv_block(in_channels=input_ch, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Average pooling 
        self.AvgPool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.flatten = torch.nn.Flatten()
        # 2nd layer (MLP) 
        # conv_output = 7*7*20= 980
        self.MLP2 = torch.nn.Linear(in_features=980, out_features=class_num, bias=True)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        layer1 = self.conv1(x)
        layer1 = self.pool1(layer1)
        Pool_layer = self.AvgPool(layer1)
        Pool_layer = self.flatten(Pool_layer)
        Output_layer = self.MLP2(Pool_layer) 
        return self.softmax(Output_layer) 

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):     
        return nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), 
            torch.nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True) 
        )

def SelfONN_1(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   

        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=75,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.BatchNorm2d(75, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=75,out_channels=56,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.BatchNorm2d(56, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        torch.nn.Flatten(),  
    
        # Output layer (MLP)  
        
        torch.nn.Linear(in_features=2744, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

# def SelfONN_1(input_ch, class_num, q_order): 
#     model = torch.nn.Sequential(   
#         # 1st layer (conv) 
#         SelfONNLayer(in_channels=input_ch,out_channels=2,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(4),
#         torch.nn.Tanh(),
#         # 2nd layer (conv)
#         SelfONNLayer(in_channels=2,out_channels=4,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
#         torch.nn.MaxPool2d(4),  
#         torch.nn.Tanh(), 
#         # flatten 
#         torch.nn.Flatten(),  
#         # Output layer (MLP)  
#         torch.nn.Linear(in_features=576, out_features=class_num, bias=True),  
#         torch.nn.LogSoftmax(dim=1)
#     ) 
#     #
#     reset_fn = reset_function_generic 
#     model.apply(reset_fn) 
#     return model 


def SelfONN_2(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=784, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_2_dense(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(4),  
        torch.nn.Tanh(),  
        # Output layer (Self-MLP) 
        SelfONNLayer(in_channels=16,out_channels=class_num,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast'), 
        SqueezeLayer(),
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 



def SelfONN_3(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast',dropout=0.2),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        #torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)
        torch.nn.Dropout(p=0.2, inplace=False),  
        torch.nn.Linear(in_features=1568, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

def SelfONN_3_SelfDense(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),  
        torch.nn.Tanh(), 
        # Output layer (Self-MLP) 
        SelfONNLayer(in_channels=16,out_channels=class_num,kernel_size=3,stride=1,padding=0,dilation=1,groups=1,bias=True,q=q_order,mode='fast'), 
        SqueezeLayer(),
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 



def SelfONN_4(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 


def SelfONN_5(input_ch, class_num, q_order): 
    model = torch.nn.Sequential(   
        # 1st layer (conv) 
        SelfONNLayer(in_channels=input_ch,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 2nd layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 3rd layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 4th layer (conv)
        SelfONNLayer(in_channels=8,out_channels=8,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 5th layer (conv) 
        SelfONNLayer(in_channels=8,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 6th layer (conv)
        SelfONNLayer(in_channels=16,out_channels=16,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.Tanh(), 
        # 7th layer (conv) 
        SelfONNLayer(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(2),
        torch.nn.Tanh(),
        # 8th layer (conv)
        SelfONNLayer(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1,dilation=1,groups=1,bias=True,q=q_order,mode='fast'),
        torch.nn.MaxPool2d(3),  
        torch.nn.Tanh(), 
        # flatten 
        torch.nn.Flatten(),  
        # Output layer (MLP)  
        torch.nn.Linear(in_features=512, out_features=class_num, bias=True),  
        torch.nn.LogSoftmax(dim=1)
    ) 
    #
    reset_fn = reset_function_generic 
    model.apply(reset_fn) 
    return model 

def get_pretrained_model(parentdir, model_name,ImageNet,input_ch,class_num,train_on_gpu,multi_gpu,q_order):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """
  

    if model_name == 'convit_tiny':
        model = timm.create_model("convit_tiny", pretrained=ImageNet)
        n_inputs = model.head.in_features
        model.head = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    
    if model_name == 'mobilevit_s':
        model = timm.create_model("mobilevit_s", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)
    if model_name == 'mvitv2_small':
        model = timm.create_model("mvitv2_small", pretrained=True)
        n_inputs = model.head.fc.in_features
        model.head.fc = CNN_Classifier(n_inputs, class_num)
        if not ImageNet:
            reset_fn = reset_function_generic 
            model.apply(reset_fn)

    

    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)

    return model 







# class CustomLayer(nn.Module):
#             def __init__(self,layer_idx=None,in_channels=1,out_channels=1,kernel_size=1,sampling_factor=1,optimize=True):
#                 super().__init__() 
#                 self.in_channels = in_channels 
#                 self.out_channels = out_channels
#                 self.kernel_size = kernel_size
#                 self.sampling_factor = sampling_factor
#                 self.layer_idx = layer_idx
#             def forward(self,x): 
#                 x = x.squeeze(3) 
#                 x = x.squeeze(2)  
#                 return x 
