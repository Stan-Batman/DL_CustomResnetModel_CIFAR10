# DL_CustomResnetModel_CIFAR10

Model: Modified Resnet 

Dataset: CIFAR-10


# PreRequisistes


The following python packages are required:

1. torch
2. torchsummary
3. numpy
4. tqdm
5. multiprocessing

Install them manually or use this in your python notebook: ! pip install torch torchsummary numpy tqdm multiprocessing


In Deep learning, data augmentation techniques and transforms are essential to improve the quality and quantity of data used to train models. In the case of the CIFAR10 dataset, which consists of 60,000 color images of 32x32 pixels, these techniques are particularly important.

To augment the data, we use two techniques - Random Crop and Random Horizontal Flip. Random Crop selects a random subset of the original image, while Random Horizontal Flip flips an image horizontally with a certain probability. Both of these techniques help to introduce variations in the dataset, making the model more capable of recognizing similar images with different compositions.

After applying the data augmentation techniques, we use the To Tensor transform, which converts the images into PyTorch tensors and scales them by 255. This transformation is essential because PyTorch works with tensors, and scaling the images helps to normalize the pixel values.

Lastly, we apply the Normalize transform to adjust the mean and standard deviation of the image pixels, making the images more standardized. The values of the pixels become 0.0 and 1.0, respectively.

To ensure consistency and reproducibility, we download the CIFAR10 dataset in the root directory ./data and set PyTorch's random number generator to a seed value of 17. By doing this, we get the same validation set each time, which helps us to evaluate the model's performance accurately.

Overall, applying data augmentation techniques and transforms to the CIFAR10 dataset helps to improve the quality of the data and enhance the model's performance. These techniques can also be applied to other datasets, making it an essential part of machine learning workflows.

We have trained our Custom Resnet Model on CiFAR-10 Dataset and have tried 3 different optimisers:

1. Adam

2. Adagrad

3. daDelta

AdaDelta Performed the best out of these.

LR: 0.01 


____________________________

## Files Overview

Visualization_final.ipynb: Shows graphical comparison between the different optimizer, Adam, AdaGrad, AdaDelta; with multiple learning rates.   

Visualization_final.pdf: PDF of the Visualization_final.ipynb notebook with executed cells

Adam:  
  - Adam_LR-0.01.ipynb: Shows train and test performance of the model with Adam optimizer and learning rate 0.01  
  - Adam_LR-0.001.ipynb: Shows train and test performance of the model with Adam optimizer and learning rate 0.001  
  
AdaGrad: 
  - 0.01_LR_model_adagrad.ipynb: Shows train and test performance of the model with AdaGrad optimizer and learning rate 0.01  
  - model_adagrad (0.001 LR).ipynb: Shows train and test performance of the model with AdaGrad optimizer and learning rate 0.001  
  
AdaDelta:  
  - miniproject_adaDelta_lr_0pt1_aas10120_final.ipynb: Shows train and test performance of the model with AdaDelta optimizer and learning rate 0.1  
  - miniproject_adaDelta_lr_0pt01_aas10120_final.ipynb: Shows train and test performance of the model with AdaDelta optimizer and learning rate 0.01  
  - miniproject_adaDelta_lr_0pt001_aas10120_final (1).ipynb.ipynb: Shows train and test performance of the model with AdaDelta optimizer and learning rate 0.001 
  
PDF Subfolder: Contains PDF of all uploaded Jupyter Notebooks

finalModelArchitecture.jpg: Final Model Architecture (using Netron API)

final_model1.onnx: onnx file of the model architecture

miniProjectReport_DL_final.pdf: AAAI format Project Report

miniproject_adaDelta_lr_0pt01_final-bestAccuracy.ipynb: Shows train and test performance of the model with AdaDelta optimizer and learning rate 0.01. This had the highest testing accuracy as well.

____________________________


|| Training Accuracy: 99.992% ||


____________________________

|| Testing Accuracy: 91.47%   ||

____________________________



