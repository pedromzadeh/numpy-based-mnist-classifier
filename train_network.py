from curses.panel import new_panel
from ipaddress import NetmaskValueError
from unittest import TestLoader
import numpy as np
import matplotlib.pyplot as plt
import network as nn
import load_data as ld

# hyperparameters
N_train = 40000
N_test = 5000
N_batch = 50
N_epoch = 15
lr = 10

training_data, testing_data = ld.load_data(N_train,N_test)
network_size = [784,30,10]
net = nn.Network(network_size)

cost = []
accuracy = []
for ep in range(N_epoch):
   # turn input data into mini batches
   batches = [training_data[k:k+N_batch] for k in range(0,len(training_data),N_batch)] 

   # pass all the training data through the network, one mini_batch at a time
   for mini_batch in batches:
      net.SGD(mini_batch,lr)

   cost.append(net.cost(training_data))
   accuracy.append(net.evaluate(testing_data))

   print('Epoch {}: {} accurate'.format(ep,accuracy[ep]))

plt.subplot(121)
plt.plot(range(N_epoch),accuracy)
plt.xlabel('Epoch #')
plt.ylabel('Accuracy')

plt.subplot(122)
plt.plot(range(N_epoch),cost)
plt.xlabel('Epoch #')
plt.ylabel('Cost')
plt.show()

test_imgs, test_lbs = zip(*testing_data)
fig = plt.figure(figsize=(8,8))
for s in range(1,36):
   pred_lb = np.argmax(net.feedforward(test_imgs[s-1]))
   plt.subplot(6,6,s).set_title('ELabel: {}, PLabel: {}'.format(test_lbs[s-1],pred_lb))
   if pred_lb == test_lbs[s-1]:
      plt.imshow(test_imgs[s-1].reshape(28,28),cmap='Greens')
   else:
      plt.imshow(test_imgs[s-1].reshape(28,28),cmap='Reds')
   plt.axis('off')
plt.show()
