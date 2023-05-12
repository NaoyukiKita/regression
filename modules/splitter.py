#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np


# ### <span style="color: yellow; ">Splitter For K-Fold Cross Validation</span>

# This class splits the Design Matrix ${\bf \Phi}$ and ${\bf Y}$ into train/test dataset. This is designed to be used in K-fold cross validation(see below).
# 
# This consists of the following methods and fields:
# 
# <span style="color: red; ">**methods**</span>
# * \_\_init\_\_(self, Phi, Y, num_splits, shuffle, seed) : *constructor*
# 
#     Store the arguments and compute the size of the test subset using `num_splits`, which denotes the number of subsets. `indices` is used for partitioning the dataset(`Phi`(${\bf \Phi}$) and `Y`(${\bf Y}$)) and is determined at random if `shuffle` is True, or in order if not.
# 
# * resetTestIndicator(self) :
# 
#     Reset `test_indicator` to 0. See <span style="color: red; ">**fields**</span> for detail.
# 
# * split(self) :
# 
#     Partition the dataset(`Phi`(${\bf \Phi}$) and `Y`(${\bf Y}$)) into train/test dataset. Each time this is called, the position of the test dataset changes.
# 
# <span style="color: red; ">**fields**</span>
# 
# * num_splits :
# 
#     the number of subsets you want to split into
# 
# * subset_size :
# 
#     the size of the test subset
# 
# * test_indicator :
# 
#     an index indicating the position of the test subset among the subsets.
# 
# * indices :
# 
#     indices determined to collect the elements of each subset. If `shuffle` is True, this will be shuffled.
# 
# * Phi, Y :
# 
#     the copy of the arguments `Phi`(${\bf \Phi}$) and `Y`(${\bf Y}$).

# In[41]:


class Splitter():
    def __init__(self, Phi: np.ndarray, Y: np.ndarray, num_splits: int, shuffle: bool=False, seed: int=None):
        self.num_splits = num_splits
        self.subset_size = int(Phi.shape[0] / num_splits)
        self.test_indicator = 0

        if shuffle:
            if seed is not None: np.random.seed(seed)
            self.indices = np.random.permutation(Phi.shape[0])
        else:
            self.indices = np.arange(Phi.shape[0])
        
        self.Phi = Phi.copy()
        self.Y = Y.copy()
        
        return
    
    def resetTestIndicator(self):
        self.test_indicator = 0

        return
    
    def split(self):
        if self.test_indicator == self.num_splits:
            print('WARNING: All of subsets will be used for test(self.test_indicator == self.num_splits).', end='')
            print('reset self.test_indicator to 0')
            self.resetTestIndicator()
            
        test_start = self.subset_size * self.test_indicator
        if self.test_indicator != self.num_splits-1:
            test_stop = test_start + self.subset_size
        else:
            test_stop = self.Phi.shape[0]

        test_indices = self.indices[test_start:test_stop]
        train_indices = np.concatenate((self.indices[:test_start], self.indices[test_stop:]))

        self.test_indicator = self.test_indicator + 1

        return self.Phi[train_indices], self.Phi[test_indices], self.Y[train_indices], self.Y[test_indices]


# #### <span style="color: yellow; ">Module Test</span>

# As a test, we generate a toy sample:
# $$
# {\bf \Phi} = 
#     \left[\begin{array}{c}
#         0 & 1 & 2 \\
#         1 & 2 & 3 \\
#         \vdots & \vdots & \vdots \\
#         5 & 6 & 7 \\
#     \end{array}\right]
# 
# {\bf Y} =
#     \left[\begin{array}{c}
#         3 \\
#         4 \\
#         \vdots \\
#         8 \\
#     \end{array}\right]
# $$

# In[42]:


if __name__ == '__main__':
    num_samples = 6
    Phi = []
    Y = []
    for row in range(num_samples):
        Phi.append(list(range(row, row+3)))
        Y.append([row+3])
    
    Phi = np.array(Phi)
    Y = np.array(Y)
    print(Phi)
    print(Y)


# In[43]:


if __name__ == '__main__':
    num_splits = 3
    splitter = Splitter(Phi, Y, num_splits)

    for iter in range(num_splits):
        Phi_train, Phi_test, Y_train, Y_test = splitter.split()
        print('Phi_train')
        print(Phi_train)
        print('Phi_test')
        print(Phi_test)
        print('Y_train')
        print(Y_train)
        print('Y_test')
        print(Y_test)
        print()
    


# In[44]:


if __name__ == '__main__':
    num_splits = 3
    splitter = Splitter(Phi, Y, num_splits, shuffle=True, seed=0)
    Phi_train, Phi_test, Y_train, Y_test = splitter.split()
    print('Phi_train')
    print(Phi_train)
    print('Phi_test')
    print(Phi_test)
    print('Y_train')
    print(Y_train)
    print('Y_test')
    print(Y_test)
    print()


# In[45]:


if 'get_ipython' in globals():
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '*.ipynb'])
    print('Saved as splitter.py')

