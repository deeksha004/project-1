import warnings
#warnings.filterwarnings("ignore")
import os
import numpy as np
import time
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
#import torch.backends.cudnn as cudnn
import pandas as pd
import torch.optim as optim
import random
import sys
import glob
import matplotlib.pyplot as plt
#import segmentation_models_pytorch as smp
import sys
#sys.path.insert(0, 'over9000/')

#os.system(f"""git clone https://github.com/mgrankin/over9000.git""")
#from ralamb import Ralamb
#from radam import RAdam
#from ranger import Ranger
#from lookahead import LookaheadAdam
#from over9000 import Over9000
from tqdm.notebook import tqdm
from tqdm import tqdm_notebook as tqdm

#For Transformations
#import tifffile as tiff
#from torch.utils.data import Dataset, DataLoader, sampler
#import albumentations as aug
#from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose,Cutout, GaussNoise, RandomRotate90, Transpose, RandomBrightnessContrast, RandomCrop)
#from albumentations.pytorch import ToTensor

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

def double_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
  double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
  return double_conv
def conv(in_channels, out_channels, kernel_size, stride=1, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def up_conv():
   uplayers = []
   upconv_layer=nn.Upsample(scale_factor=2)
   uplayers.append(upconv_layer)
   return nn.Sequential(*uplayers)

class U_NET_MOD1(nn.Module):

    def __init__(self, n_classes=1):
        super().__init__()
                
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = double_conv(256 + 512, 256)
        self.conv_up2 = double_conv(128 + 256, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        
    def forward(self, x):
        # Batch - 1d tensor.  N_channels - 1d tensor, IMG_SIZE - 2d tensor.
        # Example: x.shape >>> (10, 3, 256, 256).
        
        conv1 = self.conv_down1(x)  # <- BATCH, 3, IMG_SIZE  -> BATCH, 64, IMG_SIZE..
        x = self.maxpool(conv1)     # <- BATCH, 64, IMG_SIZE -> BATCH, 64, IMG_SIZE 2x down.
        conv2 = self.conv_down2(x)  # <- BATCH, 64, IMG_SIZE -> BATCH,128, IMG_SIZE.
        x = self.maxpool(conv2)     # <- BATCH, 128, IMG_SIZE -> BATCH, 128, IMG_SIZE 2x down.
        conv3 = self.conv_down3(x)  # <- BATCH, 128, IMG_SIZE -> BATCH, 256, IMG_SIZE.
        x = self.maxpool(conv3)     # <- BATCH, 256, IMG_SIZE -> BATCH, 256, IMG_SIZE 2x down.
        x = self.conv_down4(x)      # <- BATCH, 256, IMG_SIZE -> BATCH, 512, IMG_SIZE.
        x = self.upsample(x)        # <- BATCH, 512, IMG_SIZE -> BATCH, 512, IMG_SIZE 2x up.
        
        #(Below the same)                                 N this       ==        N this.  Because the first N is upsampled.
        x = torch.cat([x, conv3], dim=1) # <- BATCH, 512, IMG_SIZE & BATCH, 256, IMG_SIZE--> BATCH, 768, IMG_SIZE.
        
        x = self.conv_up3(x) #  <- BATCH, 768, IMG_SIZE --> BATCH, 256, IMG_SIZE. 
        x = self.upsample(x)  #  <- BATCH, 256, IMG_SIZE -> BATCH,  256, IMG_SIZE 2x up.   
        x = torch.cat([x, conv2], dim=1) # <- BATCH, 256,IMG_SIZE & BATCH, 128, IMG_SIZE --> BATCH, 384, IMG_SIZE.  

        x = self.conv_up2(x) # <- BATCH, 384, IMG_SIZE --> BATCH, 128 IMG_SIZE. 
        x = self.upsample(x)   # <- BATCH, 128, IMG_SIZE --> BATCH, 128, IMG_SIZE 2x up.     
        x = torch.cat([x, conv1], dim=1) # <- BATCH, 128, IMG_SIZE & BATCH, 64, IMG_SIZE --> BATCH, 192, IMG_SIZE.  
        
        x = self.conv_up1(x) # <- BATCH, 128, IMG_SIZE --> BATCH, 64, IMG_SIZE.
        
        out = self.last_conv(x) # <- BATCH, 64, IMG_SIZE --> BATCH, n_classes, IMG_SIZE.
        out = torch.sigmoid(out)
        
        return out
        
def model_predict(model, img_path):
	dim = (256, 256)
	thresh=0.5
	input_img=img_path
	input_img = plt.imread(img_path)[:,:,:3]
	h=input_img.shape[0]
	w=input_img.shape[1]
	#change to BGR
	img = cv2.resize(input_img, dim, interpolation = cv2.INTER_NEAREST)
	img = np.array(img)
	img = img.transpose((2, 0, 1))
	img = img/255
	# Normalize based on the preset mean and standard deviation
	img[0] = (img[0] - 0.485)/0.229
	img[1] = (img[1] - 0.456)/0.224
	img[2] = (img[2] - 0.406)/0.225
	# Add a fourth dimension to the beginning to indicate batch size
	img = img[np.newaxis,:]
	image = torch.from_numpy(img)
	image = image.float()

	images = image.to(device)
	preds = model(images)
	preds=preds.squeeze(0).squeeze(0).detach().cpu().numpy()
	preds=(preds > thresh).astype('uint8')*255
	pred_mask=cv2.resize(preds,(w,h))
	gs_percentage = round((np.sum(pred_mask==255)/(h*w))*100, 3)
	return pred_mask, gs_percentage
 

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
    
@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		# Get the file from post request
		f=request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		#predict the image
		pred_mask, gs_percentage=model_predict(model, file_path)
		result=str(gs_percentage)+"%"
		return result
	return None
if __name__ == '__main__':
	model=U_NET_MOD1(1)
	device=torch.device("cpu")#cuda:0")
	model_path="models/"
	model_name="Unet_Mod1_bs_32_p256_lrplt_IOU_Adam_150_0.001_best_loss_62.pth"

	#loadmodel
	checkpoint = torch.load(model_path+model_name, map_location=device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	#pred_mask, gs_percentage=model_predict(model, "GOPR2274_frame00093.jpg")#file_path)
	#plt.imshow(pred_mask)
	#plt.show()
	#print(str(gs_percentage)+"%")
	
	app.run(debug=True)
    
