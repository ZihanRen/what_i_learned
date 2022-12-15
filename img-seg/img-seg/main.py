from model import DCNN_multiclass, DCNN_binaryclass
from torch import nn
from Segdataset import Segdataset
import os 
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


batch_size = 30
lr = 1e-03
n_epochs = 30
device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running device available: {}'.format(torch.cuda.get_device_name(0)))

# dataset loading and initialization
PATH = os.path.join("d:\\","data","img-seg","seep_detection","train_images_256")
PATH_m = os.path.join("d:\\","data","img-seg","seep_detection","train_masks_256")
names_msk = os.listdir(PATH_m)
names_img = os.listdir(PATH)

names_img_train,names_img_test,names_msk_train,names_msk_test = train_test_split(
  names_img, names_msk,
	test_size=0.2, random_state=0
  )

segtrain = Segdataset(PATH,PATH_m,names_img_train,names_msk_train)
train_data_loader = DataLoader(segtrain,batch_size=batch_size,shuffle=True)
segtest = Segdataset(PATH,PATH_m,names_img_test,names_msk_test)
test_data_loader = DataLoader(segtest,batch_size=batch_size,shuffle=True)

# set up loss functions and load neural network models
criterion_multi = nn.CrossEntropyLoss()
cnn_multi = DCNN_multiclass().to(device)
criterion_binary = nn.CrossEntropyLoss()
cnn_binary = DCNN_binaryclass().to(device)

optimizer_multi = optim.Adam(cnn_multi.parameters(),lr=lr)
optimizer_binary = optim.Adam(cnn_binary.parameters(),lr=lr)

train_step = len(segtrain)/batch_size
test_step = len(segtest)/batch_size


def convert_msk(msk_tensor):
    # convert multi class msk to binary msk
    # 1 contains seep
    msk_b = msk_tensor.clone()
    msk_b[msk_b>0] = 1
    return msk_b

# beginnning training
t_losses = []
val_losses = []
bt_losses = []
bval_losses = []

# beginning training
print('Training begin: ...')
import time
start = time.time()
torch.autograd.set_detect_anomaly(True)
for epoch in range(n_epochs):
  train_loss_per_epoch = 0
  test_loss_per_epoch = 0
  binary_train_loss_per_epoch = 0
  binary_test_loss_per_epoch = 0

  
  for i, (img,msk) in enumerate(train_data_loader):
    img,msk = img.to(device),msk.to(device)
    # produce msk specific for binary classification
    msk_b = convert_msk(msk)

    ## train CNN multiclass segmentation ##
    optimizer_multi.zero_grad()
    msk_pred = cnn_multi(img)
    loss_multi = criterion_multi(msk_pred,msk)
    loss_multi.backward()
    optimizer_multi.step()
    train_loss_per_epoch += loss_multi.item()

    ## train CNN binary segmentation ##
    optimizer_binary.zero_grad()
    msk_pred_b = cnn_binary(msk_pred.detach())
    loss_binary = criterion_binary(msk_pred_b,msk_b)
    loss_binary.backward()
    optimizer_binary.step()
    binary_train_loss_per_epoch += loss_binary.item()

    ## validate training performance ##
    with torch.no_grad():
      for i, (img,msk) in enumerate(test_data_loader):
        img,msk = img.to(device),msk.to(device)
        msk_b = convert_msk(msk)

        msk_pred = cnn_multi(img)
        msk_pred_b = cnn_binary(msk_pred)

        loss_multi = criterion_multi(msk_pred,msk)
        loss_binary = criterion_binary(msk_pred_b,msk_b)

        test_loss_per_epoch += loss_multi.item()
        binary_test_loss_per_epoch += loss_binary.item()

    
  train_loss_per_epoch /= train_step
  test_loss_per_epoch /= test_step
  binary_train_loss_per_epoch /= train_step
  binary_test_loss_per_epoch /= test_step


  ## Output training stats ##
  print('\n')
  print('Training loss at epoch {} : {:.5f}'.format(epoch,train_loss_per_epoch))
  print('Testing loss at epoch {} : {:.5f}'.format(epoch,test_loss_per_epoch))
  print('Binary training loss at epoch {} : {:.5f}'.format(epoch,binary_train_loss_per_epoch))
  print('Binary testing loss at epoch {} : {:.5f}'.format(epoch,binary_test_loss_per_epoch))  

  t_losses.append(train_loss_per_epoch)
  val_losses.append(test_loss_per_epoch)
  bt_losses.append(binary_train_loss_per_epoch)
  bval_losses.append(binary_test_loss_per_epoch)  

print("Total training time takes approximately: ", (time.time() - start)/60, "minutes")

# visualize performance
os.chdir('submit-cgg/img-seg')
f = plt.figure(figsize=(10,5))
plt.title("Multi classification training history")
plt.plot(t_losses,'b',label="multi class training loss")
plt.plot(val_losses,'b',label="multi class validation loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
f.savefig('train-hist-multi.png')

f = plt.figure(figsize=(10,5))
plt.title("Binary classification training history")
plt.plot(bt_losses,'r',label="binary training loss")
plt.plot(bval_losses,'r',label="binary validation loss")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
f.savefig('train-hist-binary.png')

def plot_test_mask(index,true, pred, pred_m):
    true_m = true
    true_m[true_m>0] = 1
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    
    ax[0,0].imshow(true)
    ax[0,0].set_title("True multi Mask")
    
    ax[0,1].imshow(pred)
    ax[0,1].set_title("Pred multi Mask")

    ax[1,0].imshow(true_m)
    ax[1,0].set_title("True binary Mask")
    
    ax[1,1].imshow(pred_m)
    ax[1,1].set_title("Pred binary Mask")

    fig.savefig('compare-sample{}.png'.format(index))

    plt.show()

def tensor_process(image_tensor):
  # convert image tensor to numpy array
    image_unflat = image_tensor.detach().cpu()
    image_numpy = image_unflat.numpy()

    return image_numpy

def prob_to_class(arr):
  # convert prediction results to real classes indices
  arr = np.argmax(arr,axis=1)
  return arr

with torch.no_grad():
  img_sample,msk_sample = next(iter(test_data_loader))
  img_sample,msk_sample = img_sample.to(device),msk_sample.to(device)
  msk_pred = cnn_multi(img_sample)
  msk_pred_b = cnn_binary(msk_pred)

# process tensor
msk_pred = F.softmax(msk_pred,dim=1)
msk_pred_b = F.softmax(msk_pred_b,dim=1)

msk_sample,msk_pred, msk_pred_b = tensor_process(msk_sample),tensor_process(msk_pred),tensor_process(msk_pred_b)
msk_pred, msk_pred_b = prob_to_class(msk_pred), prob_to_class(msk_pred_b)

for idx in range(10):
    print("\n\n\nSample {}".format(idx))
    plot_test_mask( idx,msk_sample[idx], msk_pred[idx],msk_pred_b[idx] )


