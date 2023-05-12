#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# ### <span style="color: yellow; ">Visualizer</span>

# For better understanding, we should visualize the results.
# 
# This consists of the following methods and fields:
# 
# <span style="color: red; ">**methods**</span>
# 
# * visualize(cls, predicted, target) : *class method*
# 
#     Plot the predicted data, and scatter the target data. If the data is multi-dimentional, visualize each of dimention.
#     x-axis is indices of the data, and y-axis is values of it.
# 
# <span style="color: red; ">**fields**</span>
# 
# \*\* None \*\*

# In[21]:


class Visualizer():
    @classmethod
    def visualize(cls, predicted: np.ndarray, target: np.ndarray):
        for column in range(predicted.shape[1]):
            y_hat = predicted[:, column].T
            y = target[:, column].T
            
            plt.plot(np.arange(predicted.shape[0]), y_hat, label='predicted', color='red')
            plt.scatter(np.arange(predicted.shape[0]), y, label='target')
            plt.title(str(column) + '\'th elements of output')
            plt.legend()
            plt.show()


# #### <span style="color: yellow; ">Module Tester</span>

# In[22]:


if __name__ == '__main__':
    num_samples = 30
    output_size = 2
    predicted = np.zeros((num_samples, output_size))
    target = np.zeros((num_samples, output_size))

    np.random.seed(0)
    for row in range(num_samples):
        predicted[row][0] = np.sin(row / 15 * np.pi)
        predicted[row][1] = np.cos(row / 15 * np.pi)

        target[row][0] = np.sin(row / 15 * np.pi) + np.random.rand() * 0.6 - 0.3
        target[row][1] = np.cos(row / 15 * np.pi) + np.random.rand() * 0.6 - 0.3
    
    Visualizer.visualize(predicted, target)


# In[23]:


if 'get_ipython' in globals():
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '*.ipynb'])
    print('Saved as visualizer.py')

