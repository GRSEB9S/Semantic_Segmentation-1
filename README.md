# Semantic Segmentation
### Introduction
In this project, the pixels of a road are labeled using a Fully Convolutional Network (FCN).

### Sample Output

<img src=./runs/40.png width="420" height= "150"> <img src=./runs/um_000040.png width="420" height= "150">
<img src=./runs/42.png width="420" height= "150"> <img src=./runs/um_000042.png width="420" height= "150">
<img src=./runs/44.png width="420" height= "150"> <img src=./runs/um_000044.png width="420" height= "150">
<img src=./runs/47.png width="420" height= "150"> <img src=./runs/um_000047.png width="420" height= "150">
<img src=./runs/49.png width="420" height= "150"> <img src=./runs/um_000049.png width="420" height= "150">

##### Dataset
Downloaded the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extracted the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```

 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers.
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to the loss function, otherwise regularization is not properly implemented.
