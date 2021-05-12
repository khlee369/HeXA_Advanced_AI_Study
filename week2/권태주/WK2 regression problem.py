#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt
import random as ran
import scipy as sp

X = np.linspace(-4.5, 4.5, 10)
Y = np.array([0.9819, 0.7973, 1.9737, 0.1838, 1.3180, -0.8361, -0.6591, -2.4701, -2.8122, -6.2512])

plt.figure(figsize=(10, 6))
plt.plot(X, Y, 'o')
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.xlim(-5, 5)
plt.ylim(-7, 3)
plt.show()

W0 = np.random.rand(1, 10)
W1 = np.random.rand(1, 10)
W2 = np.random.rand(1, 10)

def hypothesis(x):
    return np.matmul(W2, x**2) + np.matmul(W1, x) + W0

def cost(x, y):
    return np.mean(np.sum(np.square(y - hypothesis(x))))

print(cost(X, Y))

print('cost 1 : ', cost(X, Y))
learning_rate = 0.00001


           
for i in range(501):
    W_0 = W0 - learning_rate * np.mean(np.sum(hypothesis(X) - Y))
    W_1 = W1 - learning_rate * np.mean(np.sum(hypothesis(X) - Y) * (X))
    W_2 = W2 - learning_rate * np.mean((np.sum(hypothesis(X) - Y) * (X**2)))
    
    W0 = W_0
    W1 = W_1
    W2 = W_2
    
    print(cost(X, Y))
print(hypothesis(X))
print(Y)


# In[ ]:




