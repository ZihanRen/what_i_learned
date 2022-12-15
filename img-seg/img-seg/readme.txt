Segdataset.py: module to load seep datasets
model.py: DCNNs, which is consisted by 2 parts: multiclass and binaryclass. They are designed for different objectives. 

main.py: main scripts for running image detection task. There are 2 neural networks training at the same time: 
DCNN_multiclass is trained to classify pixels to 7 classes of seeps and non-seep (0). DCNN_binaryclass is trained to classify pixels only to binary classes: seep and non-seep. While training, DCNN_binaryclass will take the outputs of DCNN_multiclas as input and detach computational graph of DCNN_multiclas.