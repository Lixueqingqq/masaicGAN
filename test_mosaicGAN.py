import torch
import time
import os
from network import NetUskip
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import PIL
from config import opt
from utils import setNoise, learnedWN
from prepareTemplates import getImage


nz=30
nDep=4
zGL=20
zLoc=10
zPeriodic=0
skipConnections = True
Ubottleneck = -1
textureScale=1.0
contentScale=1.0

def ganGeneration(content, noise,templates=None, bVis = False):
    x = netMix(content,noise)
    if bVis:
        return x,0,0,0
    return x

model_path = '/home/moyan/service/famos_master/results/flame/ying/2019-06-13_16-02-44/netG_epoch_74_fc1.0_ngf80_ndf80_dep4-4_cLoss0.pth'
test_img_path = 'samples/test'
out_put_path = 'result'
if not os.path.exists(out_put_path):
    os.makedirs(out_put_path)

imageSize = 256
batchSize = 16

N=0
ngf = 80
ndf = 80
netMix =NetUskip(ngf, nDep, nz, bSkip=skipConnections, nc=3, ncIn=3, bTanh=True, Ubottleneck=Ubottleneck)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device",device)

netMix.load_state_dict(torch.load(model_path))
netMix.to(device)
netMix.eval()
files = os.listdir(test_img_path)
for f in files:
    if f == 'ying.jpg':
        t0 = time.time()
        print('picture_name:',f)
        im = getImage(os.path.join(test_img_path,f),bDel = True)
        im = im.to(device)
        mask = getImage('samples/ying_maskk_dilated.png',bDel = True)
        mask = mask.to(device)
        im1=im
        im = torch.mul(im,mask)
        
        fixnoise2 = torch.FloatTensor(1,nz, im.shape[2] // 2 ** nDep, im.shape[3] // 2 ** nDep)
        fixnoise2 = fixnoise2.to(device)
        fixnoise2=setNoise(fixnoise2)

        '''    
        else:
            if False:
                print('false')
                drift=(fixnoise2*1.0).uniform_(-1, 1)
                fixnoise2[:, zGL:fixnoise2.shape[1]-zPeriodic]+=0.05*drift[:, zGL:fixnoise2.shape[1]-zPeriodic]
        '''
        fakebig =ganGeneration(im, fixnoise2)
        print(fakebig.shape)
        fakebig = fakebig


        
        print(f,"test image size", im.shape,"inference time", -t0+time.time())
        vutils.save_image(fakebig, '%s/%s' % (out_put_path,f), normalize=True)

