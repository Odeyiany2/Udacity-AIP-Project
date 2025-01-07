# **Udacity-AIP-Project**
This is my final project submission for the Udacity's AWS Nanodegree program -- AI Programming with Python. The goal is to build an AI application that can predict the class of a flower. The dataset provided was split into three parts: [train, valid and test](https://www.kaggle.com/datasets/yousefmohamed20/oxford-102-flower-dataset) The training data has 102 categories of flowers.  

## **Deliverables**

The project is divided into two parts:

* **Part One:** This aspects builds the neural network using the one pytorch's pretrained vision model, [Efficientnet](https://pytorch.org/vision/stable/models/efficientnet.html), in a jupyter notebook. Deliverables in this part include:
  
   - Training Data Augmentation: used torchvision transforms to augment the training data with random scaling, rotations, mirroring, and/or cropping.
   - Data Normalization: the training, validation, and testing data were appropriately cropped and normalized.
   - Data Loading: the data for each set (train, validation, test) is loaded with torchvision's ImageFolder.
   - Data Batching: the data for each set is loaded with torchvision's DataLoader.
   - Petrained Network: the pretrained network, efficientnet_b0 is loaded from torchvision.models and the parameters are frozen.
   - Feedforward Classifier: I defined a new feedforward network as a classifier using the features as input.
   - Loading Checkpoints(function): a function that successfully loads a checkpoint and rebuilds the model.
   - Image Processing: `The process_image` function successfully converts a PIL image into an object that can be used as input to a trained model.
   - Class Prediction: The `predict` function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image.

* **Part Two:** This part turns the development notebook into a command line application.  Deliverables in this part include:
  
     - Training a network: `train.py` successfully trains a new network on a dataset of images.
     - Model architecture: The `training script` allows users to choose from at least two different architectures available from torchvision.models.
     - Model hyperparameters: The `training script` allows users to set hyperparameters for learning rate, number of hidden units, and training epochs.
     - Training with GPU: The `training script` allows users to choose training the model on a GPU
     - Predicting classes: The `predict.py` script successfully reads in an image and a checkpoint then prints the most likely image class and it's associated probability.
     - Top K classes: The `predict.py` script allows users to print out the top K classes along with associated probabilities.
     - Displaying class names: The `predict.py` script allows users to load a JSON file that maps the class values to other category names.
     - Predicting with GPU: The `predict.py` script allows users to use the GPU to calculate the predictions


## **Frameworks**
1. **Python**
2. **Pytorch** 
