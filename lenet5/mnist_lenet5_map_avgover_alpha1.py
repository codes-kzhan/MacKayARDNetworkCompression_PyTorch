
# coding: utf-8

# In[1]:
import torch 
torch.cuda.set_device(1)

import sys
print(sys.executable)
import torch
print (torch.__version__)
import torchvision
print(torchvision.__version__)


# In[2]:


import argparse
parser=argparse.ArgumentParser()
FLAGS=parser.parse_args(args=[])
FLAGS.batchsize=128
FLAGS.w_epoch=1000
FLAGS.w_printepoch=1
FLAGS.init_lr=1e-3
FLAGS.lrchange=0.9
FLAGS.BreakBasedLr_lr=1e-8
FLAGS.init_alpha=1

FLAGS.minvalid_avgover=5

FLAGS.save_name='mnist_lenet5_map_avgover_alpha1'


# # dataset

# In[3]:


import torchvision.datasets
import torchvision.transforms

transform1=torchvision.transforms.ToTensor()
transform2=lambda x: 2*(x-0.5)
transform12=torchvision.transforms.Compose([transform1,transform2])
datasets_mnist_train=torchvision.datasets.MNIST('../data', train=True, download=False, transform=transform12)
datasets_mnist_test=torchvision.datasets.MNIST('../data', train=False, transform=transform12)


# In[4]:


print('train size %d'% len(datasets_mnist_train.train_data) )
print(' test size %d'% len(datasets_mnist_test.test_data) )
print('image size',datasets_mnist_train.train_data[0].size() )
N = 60000.  # number of data points in the training set


# In[5]:


import torch.utils.data.dataset
import torch.utils.data
import torch


test_loader=torch.utils.data.DataLoader(datasets_mnist_test,batch_size=FLAGS.batchsize )
N=50000
train_subset=torch.utils.data.dataset.Subset(datasets_mnist_train,range(N))
valid_subset=torch.utils.data.dataset.Subset(datasets_mnist_train,range(N,60000))
train_loader=torch.utils.data.DataLoader(train_subset, batch_size=FLAGS.batchsize,shuffle=True) 
valid_loader=torch.utils.data.DataLoader(valid_subset, batch_size=FLAGS.batchsize )


# # utils

# In[6]:


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# In[7]:


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[8]:


class History(object):
    def __init__(self):
        self.loss=[]
        self.regloss=[]
        self.totalloss=[]
        self.testerr=[]  
        self.validerr=[]


# # model

# In[9]:


import torch.nn
import torch
import torch.autograd
import math
import random
import numpy
import copy
import operator

random.seed(1)
numpy.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


# In[10]:


class ARDConv2d(torch.nn.modules.Module):
    def __init__(self, in_channels, out_channels,initalpha):
        super(ARDConv2d, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, 5,5))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        torch.nn.init.kaiming_normal_(self.weights, mode='fan_in')
        self.bias.data.fill_(0)
        self.dim = out_channels
        self.alpha=torch.cuda.FloatTensor(self.dim)  
        self.alpha.fill_(initalpha)      
        
    def forward(self, input_):# batchsize,inchannel, 28x28
        output = torch.nn.functional.conv2d(input_, self.weights , self.bias )    
        return output
        
    def regularization(self):  
        reg=torch.sum(self.weights.pow(2)*self.alpha.view(-1,1,1,1))
        return 0.5*reg


# In[11]:


class ARDDense(torch.nn.modules.Module):
    def __init__(self, in_features, out_features,initalpha):
        super(ARDDense, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor( out_features,in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))        
        torch.nn.init.kaiming_normal_(self.weights, mode='fan_in')
        self.bias.data.fill_(0)
        self.dim = in_features
        self.alpha=torch.cuda.FloatTensor(self.dim)  
        self.alpha.fill_(initalpha)        
            
    def forward(self, inputx):    
        o=torch.nn.functional.linear(inputx, self.weights,self.bias)    
        return o
    
    def regularization(self): 
        reg=torch.sum(self.weights.pow(2)*self.alpha)
        return 0.5*reg


# In[12]:


class LeNet5_MNIST(torch.nn.Module):
    def __init__(self):
        super(LeNet5_MNIST, self).__init__()        
        convs=[ARDConv2d(1,20,FLAGS.init_alpha),torch.nn.ReLU(), torch.nn.MaxPool2d(2),
               ARDConv2d(20,50,FLAGS.init_alpha),torch.nn.ReLU(), torch.nn.MaxPool2d(2) ]
        self.convs=torch.nn.Sequential(*convs).cuda()         
        fcs = [ARDDense(800, 500,FLAGS.init_alpha), torch.nn.ReLU(),
               ARDDense(500, 10,FLAGS.init_alpha)]
        self.fcs = torch.nn.Sequential(*fcs).cuda()
             
    def forward(self, x):
        o = self.convs(x)
        o = o.view(o.size(0), -1)
        o=self.fcs(o)
        return o
    
    def regularization(self):
        regularization = 0.0
        for m in self.modules():
            if isinstance(m, ARDDense) or isinstance(m, ARDConv2d):
                regularization += m.regularization() 
        return regularization


# In[13]:


model = LeNet5_MNIST()
model = model.cuda()
lossfunc=torch.nn.CrossEntropyLoss()


# # run

# In[14]:


def oneThroughdataset_test(dataloader ):    
    loss_avg = AverageMeter()
    predictionErr_avg = AverageMeter()
    model.eval()       
    for ibatch, (input_, target) in enumerate(dataloader):
        target = target.cuda(async=True)
        input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)
        loss = lossfunc(output, target_var)       
        acc = accuracy(output.data, target, topk=(1,))[0]    
        loss_avg.update(loss.data.item(), input_.size(0))
        predictionErr_avg.update(100 - acc.item(), input_.size(0))
    return loss_avg.avg, predictionErr_avg.avg


# In[15]:


def oneThroughdataset_train(optimizer):   
    loss_avg = AverageMeter()
    regloss_avg = AverageMeter()
    totalloss_avg = AverageMeter()
    predictionErr_avg = AverageMeter()
    model.train()
    for ibatch, (input_, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input_ = input_.cuda()
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)
        output = model(input_var)        
        loss = lossfunc(output, target_var)
        regloss=model.regularization()/N
        totalloss = loss +  regloss   
        totalloss = totalloss.cuda()
        acc = accuracy(output.data, target, topk=(1,))[0]   
        
        loss_avg.update(loss.data.item(), input_.size(0))
        regloss_avg.update(regloss.data.item(), input_.size(0))
        totalloss_avg.update(totalloss.data.item(), input_.size(0))
        predictionErr_avg.update(100 - acc.item(), input_.size(0))
        
        optimizer.zero_grad()
        totalloss.backward()
        optimizer.step()
          
    return loss_avg.avg, regloss_avg.avg,  totalloss_avg.avg, predictionErr_avg.avg


# In[16]:


import numpy
def z_step(history):
    last1,last2=1e+10,1e+10
    
    historyValidErr=[]
    
    bestavgvalid,storevalid,storetest,storeepoch=100,100,100,-1
    
    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': FLAGS.init_lr}])
    learningrate=FLAGS.init_lr
    for w_iter in range(FLAGS.w_epoch):    
        meanloss,meanregloss,meantotalloss,predictionerr  =oneThroughdataset_train(optimizer)
        validloss,validerr=oneThroughdataset_test(valid_loader)
        testloss,testerr=oneThroughdataset_test(test_loader)
            
        history.loss.append( meanloss )
        history.regloss.append( meanregloss)
        history.totalloss.append(meantotalloss )
        history.testerr.append(testerr)
        history.validerr.append(validerr)
        
        for tmpd in range(len(historyValidErr)-1):
            historyValidErr[tmpd]=historyValidErr[tmpd+1]
        if len(historyValidErr)<FLAGS.minvalid_avgover:
            historyValidErr.append(validerr)
        else:
            historyValidErr[-1]=validerr
            
        
        if  numpy.mean(historyValidErr)<bestavgvalid:
            bestavgvalid=numpy.mean(historyValidErr)
            storevalid=validerr
            storetest=testerr
            storeepoch=w_iter
            
            state=dict()
            state['statedict']=model.state_dict()
            state['alpha']=FLAGS.init_alpha
            state['testerr']=testerr
            state['validerr']=validerr
            state['witer']=w_iter
            torch.save(state, '{}-statedict-bestvalid.ckpt'.format(FLAGS.save_name) )   

        print('w {} {:.1e} err%Trn vld test={:.2f} {:.2f} {:.2f} lr={:.1e}'.format(w_iter,
                   meantotalloss,predictionerr,validerr,testerr,learningrate),end=' ')
        print('bstAvgVld vld test iter={:.3f} {:.2f} {:.2f} {}'.format(bestavgvalid,storevalid,storetest,storeepoch))
        
        
        if  (last2-last1)/last2<0.001 and (last1-meantotalloss)/last1<0.001:
            learningrate=learningrate* FLAGS.lrchange 
            optimizer.param_groups[0]['lr']=learningrate

        if learningrate < FLAGS.BreakBasedLr_lr:
            break
            
        if meantotalloss==0:
            break
            
        last2=last1
        last1=meantotalloss 
    return history,storetest,storeepoch


# In[17]:


history=History()
history,besttesterr,bestatepoch=z_step(history)
torch.save(history, '{}-history.ckpt'.format(FLAGS.save_name) )

