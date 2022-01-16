from cmd import Cmd
from sys import dont_write_bytecode
from tkinter import Y
from turtle import backward
from xmlrpc.server import DocCGIXMLRPCRequestHandler
import zlib
import numpy as np

def sigmoid(z):
   '''Returns the sigmoid of `z`.'''
   return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
   '''Returns d(sigmoid)/dz evaluated at z.'''
   return np.exp(-z) * sigmoid(z)**2

class Network():
   '''Basic elements to define a feedforward classifier with probabilistic output layer.'''

   def __init__(self,network_size):
      '''`network_size` is a list with format [# neurons, ..., # neurons]; that is, (# layers, # neurons/layer).'''
      self.network_size = network_size
      self.num_of_layers = len(network_size)
      self.weights = [np.random.randn(rows,cols) for rows,cols in zip(network_size[1:],network_size[:-1])]
      self.biases = [np.random.randn(s,1) for s in network_size[1:]]
      self.a_layers = []
      self.z_layers = []

   def feedforward(self,input_data):
      '''Compute forward propogation of the input data. At every layer, you perfrom a matrix product and apply the 
      nonlinear function. Every layer's activation and sigmoid value is stored for backpropogating later.'''
      a_l = input_data
      self.a_layers.append(input_data)
      for w,b in zip(self.weights,self.biases):
         z_l = np.dot(w,a_l) + b
         a_l = sigmoid(z_l)
         self.a_layers.append(a_l)
         self.z_layers.append(z_l)

      return a_l
      
   def backward(self,x,y):
      '''Perform a feedforward so you can compute the activations per layer, to then compute the derivatives for backpropogation. This is why
      pytorch utilizes computational graphs. Note, dC_dw needs to be a list of matrices--> d_l \cdot a.T to make a matrix.
      `x`: input data
      `y`: vectorized label'''

      # must feed forward before going back

      dC_dw = [np.zeros(shape=w.shape) for w in self.weights]
      dC_db = [np.zeros(shape=b.shape) for b in self.biases]

      d_L = (self.a_layers[-1] - y) * sigmoid_prime(self.z_layers[-1])
      dC_db[-1] = d_L
      dC_dw[-1] = np.dot(d_L,self.a_layers[-2].T)

      d_l = d_L
      for l in range(2,self.num_of_layers):
         w = self.weights[-l+1]
         d_l = np.dot(w.T,d_l) * sigmoid_prime(self.z_layers[-l])
         dC_dw[-l] = np.dot(d_l,self.a_layers[-l-1].T)
         dC_db[-l] = d_l

      return dC_dw, dC_db

   def SGD(self,mini_batch,lr):
      '''Performs a stochastic gradient descent on `mini_batch`, which is an array of `M` randomly selected input examples. The total change in `self.weights` and
      `self.biases` is accumulated over the `mini_batch` and then applied at once.
      This method executes ONE pass over the data and constitutes one "correction" step. It will have to be looped over many times elsewhere when called.
      This method calls `backward()` to compute the gradients and modulates the correction by the learning rate `lr`.'''
      M = len(mini_batch)
      
      # accumulate the change as you consider each input example
      dw_cum = [np.zeros(shape=w.shape) for w in self.weights]
      db_cum = [np.zeros(shape=b.shape) for b in self.biases]
      for x,y in mini_batch:
         self.feedforward(x)
         dw_i, db_i = self.backward(x,y)  # format: weights/biases shape x # of layers
         dw_cum = [dw_cum_prev + dw_i_l for dw_cum_prev,dw_i_l in zip(dw_cum,dw_i)]
         db_cum = [db_cum_prev + db_i_l for db_cum_prev,db_i_l in zip(db_cum,db_i)]

      # update the weights and biases according to the cumulative changes
      self.weights = [w_curr - lr/M*dw for w_curr,dw in zip(self.weights,dw_cum)]
      self.biases = [b_curr - lr/M*db for b_curr,db in zip(self.biases,db_cum)]

   def cost(self,batch):
      '''Compute the cost function for this `batch` of data.'''
      M = len(batch)
      err = 0
      for x,y in batch:
         C_m = (self.feedforward(x) - y).squeeze()
         C_m = np.dot(C_m,C_m)
         err += C_m / (2*M)
      return err

   def evaluate(self,test_data):
      '''Evaluate the accuracy of the network by making predictions on `test_data`.'''
      labels = [lb for _,lb in test_data]
      preds = [np.argmax(self.feedforward(x)) for x,_ in test_data]
      success = [int(pred==lb) for pred,lb in zip(preds,labels)]
      return np.sum(success)/len(success)



