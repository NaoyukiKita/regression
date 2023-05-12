#!/usr/bin/env python
# coding: utf-8

# In[15]:


import sys

import numpy as np

from my_model import MyModel
from Phi_generator import PhiGenerator
from splitter import Splitter


# ### <span style="color: yellow; ">K-Fold Cross Validator</span>

# $P$ in the Polynomial and ${\sigma}$ in the Gaussian are pre-set parameters but we don't know the suitable values of them.
# Let us try cross validation to find the values. In this case, we don't regard `ridge-coeff` as a pre-set parameter.
# 
# This consists of the following methods and fields:
# 
# <span style="color: red; ">**methods**</span>
# 
# * \_\_init\_\_(self, X, Y) : *constructor*
# 
#     Stores `X`(${\bf X}$) and `Y`(${\bf Y}$), then instanciates Phigenerator(see `Phi_generator.ipynb` for detail).
# 
# * __validate(self, Phi, num_splits, shuffle, seed, ridge_coeff) : *private*
# 
#     Compute a validation score by K-fold validation. The model and splitter is generated inside. The score is the average of the losses in each split, so the lower score means the better performance.
# 
# * findBestParameter(self, type, candidates, num_splits, shuffle, seed, ridge_coeff) :
# 
#     Find a parameter in `candidates` which achieved the lowest validation score. Phi(${\bf \Phi}$) is generated using `type`. If `type` is 'polynomial', `candidates` must be a list of `order`. If 'gaussian', a list of `sigma`.
# 
# <span style="color: red; ">**fields**</span>
# 
# * X, Y :
# 
#     the copy of X(${\bf X}$) and Y(${\bf Y}$)
# 
# * generator :
# 
#     a instance of Phigenerator
# 
# * model :
# 
#     a instance of MyModel to be used for validation

# In[16]:


class KFoldCV():
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = X.copy()
        self.Y = Y.copy()
        self.generator = PhiGenerator(self.X)

        return

    def __validate(self, Phi: np.ndarray, num_splits: int, shuffle: bool=False,
                   seed: int=None, ridge_coeff: float=0):
        splitter = Splitter(Phi, self.Y, num_splits, shuffle, seed)
        sum_losses = 0
        for k in range(num_splits):
            Phi_train, Phi_test, Y_train, Y_test = splitter.split()
            self.model = MyModel()
            self.model.fit(Phi_train, Y_train, ridge_coeff, False)

            loss = self.model.calcLoss(self.model.predict(Phi_test), Y_test)
            sum_losses = sum_losses + loss
        
        return (sum_losses / num_splits)
    
    def findBestParameter(self, type: str, candidates: list, num_splits: int, shuffle: bool=False,
                          seed: int=None, ridge_coeff: float=0):
        best_params = None
        best_loss = sys.float_info.max
        for candidate in candidates:

            if type == 'polynomial':
                Phi = self.generator.generatePhi(type, order=candidate)
            elif type == 'gaussian':
                Phi = self.generator.generatePhi(type, sigma=candidate)
            else:
                print('ERROR: `type` must be  \'polynomial\' or \'gaussian\'. returns None')
                return None
            
            model = MyModel()
            loss = self.__validate(Phi, num_splits, shuffle, seed, ridge_coeff)

            if best_loss <= loss: continue

            best_params = candidate
            best_loss = loss
        
        return best_params, best_loss


# #### <span style="color: yellow; ">Module Test</span>

# As a test, we generate a toy sample:
# $$
# {[{\bf y}_{n}]}_{1} = {[{\bf x}_{n}]}_{1} + {[{\bf x}_{n}]}_{2} \\
# {[{\bf y}_{n}]}_{2} = {[{\bf x}_{n}]}_{2} + {[{\bf x}_{n}]}_{3} \\
# {[{\bf y}_{n}]}_{3} = {[{\bf x}_{n}]}_{3} + {[{\bf x}_{n}]}_{1} \\
# $$
# Each of elements of ${\bf X}$ is determined at random.

# In[17]:


if __name__ == '__main__':
    num_samples = 100
    input_size = 3
    np.random.seed(0)
    X = np.random.rand(num_samples, input_size)
    Y = np.zeros((num_samples, 3))
    for row in range(num_samples):
        Y[row][0] = X[row][0] + X[row][1]
        Y[row][1] = X[row][1] + X[row][2]
        Y[row][2] = X[row][2] + X[row][0]
    
    kfcv = KFoldCV(X, Y)


# We try cross validation. We can easily expect the case with $d_{p}=1$ is the best, because we didn't include any multi-dimentional terms when generating the data.

# In[18]:


if __name__ == '__main__':
    order_candidates = [1, 2, 3]
    (best_params, best_loss) = kfcv.findBestParameter('polynomial', order_candidates, num_splits=3)

    print('best_params: ' + str(best_params))
    print('best_loss: ' + str(best_loss))

    generator = PhiGenerator(X)
    Phi = generator.generatePhi('polynomial', order=best_params)

    model = MyModel()
    model.fit(Phi, Y, verbose=True)

    loss = model.calcLoss(model.predict(Phi), Y)
    print('loss: ' + str(loss))


# In[19]:


if 'get_ipython' in globals():
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '*.ipynb'])
    print('Saved as k_fold_CV.py')

