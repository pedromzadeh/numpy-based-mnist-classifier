from unittest import TestLoader
import torchvision as tv
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

def output_vec_label(indx):
   vec = np.zeros(shape=(10,1))
   vec[indx] = 1
   return vec

def load_data(N_train,N_test,view_data=False):
   '''Use `torchvision` to download the MNIST dataset. Apply ToTensor transform to go from PIL images to tensors. Then, make an iterable out
   of the dataset by wrapping it in `DataLoader`. Separate the images and the labels. Turn the images into Numpy arrays in shape (-1,1) so 
   they're input ready for the neural net, and vectorize the int labels so they can serve as the output layer of the net.'''

   # format is [(<PIL Image>, label), len] --> [(torch tensor, label), len]
   transform = tv.transforms.Compose([tv.transforms.ToTensor()])
   cwd = os.getcwd()
   train_data = tv.datasets.MNIST(root=cwd,download=True,train=True,transform=transform)
   test_data = tv.datasets.MNIST(root=cwd,download=True,train=False,transform=transform)

   # create iterables from dataset
   train_data = torch.utils.data.DataLoader(train_data,shuffle=True,batch_size=N_train)
   test_data = torch.utils.data.DataLoader(test_data,shuffle=True,batch_size=N_test)

   # Separate images and labels, and have the images in np.array format
   train_imgs, train_lbs = next(iter(train_data))
   test_imgs, test_lbs = next(iter(test_data))

   train_imgs = np.array(train_imgs).squeeze()
   train_lbs = np.array(train_lbs)
   test_imgs = np.array(test_imgs).squeeze()
   test_lbs = np.array(test_lbs)   

   # reshape the images into (-1,1) for input into the network
   # vectorize the numerical label for the output layer of the network
   training_input = [img.reshape(-1,1) for img in train_imgs]
   training_lbs = [output_vec_label(val) for val in train_lbs]
   testing_input = [img.reshape(-1,1) for img in test_imgs]
   testing_lbs = test_lbs

   print('Dimension of the training image input data: {} x {}'.format(len(training_input),training_input[0].shape))
   print('Dimension of the training label input data: {} x {}'.format(len(training_lbs), training_lbs[0].shape))

   if view_data:
      fig = plt.figure(figsize=(8,8))
      for s in range(1,36):
         plt.subplot(6,6,s).set_title('Label: {}'.format(train_lbs[s-1]))
         plt.imshow(train_imgs[s-1],cmap='Greys')
         plt.axis('off')
      plt.show()

   return list(zip(training_input,training_lbs)), list(zip(testing_input,testing_lbs))