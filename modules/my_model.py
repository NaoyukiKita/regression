#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np


# ### <span style="color: yellow; ">Regression model</span>

# This model performs a regression analysis. This consists of the following methods and fields:
# 
# <span style="color: red; ">**methods**</span>
# * calcLoss(self, predicted, Y) :
#  
#     Calculate the value of <span style="color: red; ">**the loss function**</span>, using predicted($={\bf \Phi}{\bf w}^{*}$) and Y($={\bf Y}$):
#     $$
#     {\rm Loss}({\bf w}) = \frac{1}{N}({||{\bf Y} - {\bf \Phi w}||}^{2} + \lambda {||{\bf w}||}^{2})
#     $$
# 
# 
# * predict(self, Phi) :
# 
#     Compute outputs($={\bf \Phi}{\bf w}^{*}$) corresponding to Phi($={\bf \Phi}$):
# 
# 
# * fit(self, Phi, Y, ridge_coeff, verbose) :
# 
#     Optimize the weight matrix($={\bf w^{*}}$) using Phi($={\bf \Phi}$), Y($={\bf Y}$). It also uses ridge_coeff($=\lambda$) to regularize the matrix. 
#     $$
#     {\bf w}^{*} = {({\bf \Phi}^{T}{\bf \Phi} + \lambda {\bf I})}^{-1}{\bf \Phi}^{T}{\bf Y}
#     $$
#     If verbose is true, it displays the expected function.
# 
# 
# <span style="color: red; ">**fields**</span>
# * w_star($={\bf w^{*}}$) : 
# 
#     the weight matrix to be tuned.
# 
# * ridge_coeff($=\lambda$) :
# 
#     the coefficient of a L2-Regularization term.

# In[6]:


class MyModel():
    def __init__(self):
        self.w_star = None
        self.ridge_coeff = 0
        self.fitted = False
        return
    
    def calcLoss(self, predicted: np.ndarray, target:np.ndarray):
        if not self.fitted:
            print('WARNING: This model has not been fitted.')
        residual = predicted - target
        return (np.sum(residual ** 2) + self.ridge_coeff * np.sum(self.fitted ** 2)) / target.shape[0]

    def predict(self, Phi: np.ndarray):
        if not self.fitted:
            print('WARNING: This model has not been fitted. returns None')
            return None
        return np.dot(Phi, self.w_star)
        
    def fit(self, Phi: np.ndarray, Y: np.ndarray, ridge_coeff: float=0, verbose: bool=False):
        self.ridge_coeff = ridge_coeff

        A = np.dot(Phi.T, Phi) + ridge_coeff * np.identity(Phi.shape[1])
        A_inverse = np.linalg.inv(A)
        B = np.dot(A_inverse, Phi.T)
        self.w_star = np.dot(B, Y)

        if verbose:
            print('expected function :')
            for i in range(Y.shape[1]):
                print('y_{' + str(i) + '} =')
                for j in range(Phi.shape[1]):
                    if j: print('\n\t + ', end='')
                    print(str(self.w_star[j][i]) + ' x_{' + str(j) + '}', end='')
                print()
        
        self.fitted = True
        
        return self.w_star


# #### <span style="color: yellow; ">Module Test</span>

# As a test, we generate a toy sample:
# $$
# {[{\bf y}_{n}]}_{1} = {[{\bf x}_{n}]}_{1} + {[{\bf x}_{n}]}_{2} \\
# {[{\bf y}_{n}]}_{2} = {[{\bf x}_{n}]}_{2} + {[{\bf x}_{n}]}_{3} \\
# {[{\bf y}_{n}]}_{3} = {[{\bf x}_{n}]}_{3} + {[{\bf x}_{n}]}_{1} \\
# $$
# and ${\bf \Phi} = {\bf X}$. Each of elements of ${\bf X}$ is determined at random.

# In[7]:


if __name__ == '__main__':
    num_samples = 100
    input_size = 3
    Phi = np.random.rand(num_samples, input_size)
    Y = np.zeros((num_samples, 3))
    for row in range(num_samples):
        Y[row][0] = Phi[row][0] + Phi[row][1]
        Y[row][1] = Phi[row][1] + Phi[row][2]
        Y[row][2] = Phi[row][2] + Phi[row][0]
    
    model = MyModel()
    model.fit(Phi, Y, verbose=True)

    print(model.calcLoss(model.predict(Phi), Y))


# In[8]:


if 'get_ipython' in globals():
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '*.ipynb'])
    print('Saved as my_model.py')

