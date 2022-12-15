from torch import nn
from torchinfo import summary


class DCNN_multiclass(nn.Module):
  def __init__(self,img_channel=3,img_size=256,class_num=8):
    super(DCNN_multiclass,self).__init__()
    self.img_size = img_size
    self.class_num = class_num

    self.cnn1 = self.conv_block(img_channel,32)
    self.cnn2 = self.conv_block(32,64)
    self.cnn3 = self.conv_block(64,128)
    self.cnn4 = self.conv_block(128,class_num,final_layer=True)

  def conv_block(self,c_in,c_out,kernel_size=6,padding='same',final_layer=False):
    if final_layer:
      block = nn.Sequential(
        nn.Conv2d(c_in,c_out,kernel_size,padding=padding),
          )
    else:
      block = nn.Sequential(
        nn.Conv2d(c_in, c_out,kernel_size,padding=padding),
        nn.BatchNorm2d(num_features=c_out),
        nn.ReLU()
          )      

    return block
  
  def forward(self,x):

    x = self.cnn1(x)
    x = self.cnn2(x)
    x = self.cnn3(x)
    x = self.cnn4(x)

    return x

class DCNN_binaryclass(nn.Module):
  def __init__(self,img_size=256,input_channel=8):
    super(DCNN_binaryclass,self).__init__()
    self.img_size = img_size

    self.binary_encoder = nn.Sequential(
        nn.Conv2d(input_channel,2,kernel_size=3,padding='same'),
          )
  
  def forward(self,x):
    x = self.binary_encoder(x)
    return x


if __name__ == "__main__":
  cnn_test = DCNN_multiclass()
  cnn_test_b = DCNN_binaryclass()
  summary(cnn_test,(10,3,256,256))
  summary(cnn_test_b,(10,8,256,256))

