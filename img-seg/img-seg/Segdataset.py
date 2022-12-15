#%%
import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader


class Segdataset(Dataset):
  def __init__(self,PATH,PATH_m,names,names_m,transform=None):
    # image PATH
    self.PATH = PATH
    # mask PATH
    self.PATH_m = PATH_m
    self.transform = transform
    self.names = names
    self.names_m = names_m

  def __len__(self):
    return len(self.names)

  def __getitem__(self,index):

    # load images
    img_path = os.path.join(self.PATH,self.names[index])
    image = cv2.imread(img_path)
    image = image / 255.0
    image = np.moveaxis(image,-1,0)
    
    # load masks
    imgmsk_path = os.path.join(self.PATH_m,self.names_m[index])
    imagemsk = cv2.imread(imgmsk_path,0)

    # convert data to tensor
    image_t = torch.from_numpy(image).float()
    imagemsk_t = torch.from_numpy(imagemsk)
    imagemsk_t = imagemsk_t.long()

    if self.transform is not None:
      image_t = self.transform(image_t)
   
    return (image_t, imagemsk_t)

  def img_name(self):
    return (os.listdir(self.PATH),os.listdir(self.PATH_m))


