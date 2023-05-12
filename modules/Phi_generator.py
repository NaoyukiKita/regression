#!/usr/bin/env python
# coding: utf-8

# In[47]:


import itertools
import math

import numpy as np


# ### <span style="color: yellow; ">Phi Generator</span>

# This class generates <span style="color: red; ">**the Design Matrix**</span>($={\bf \Phi}$) from ${\bf X}$.
# 
# $$
# {\bf \Phi} = 
#     \left[\begin{array}{c}
#         {\phi}_{0}({\bf x}_{1}) & {\phi}_{1}({\bf x}_{1}) & {\phi}_{2}({\bf x}_{1}) & \cdots & {\phi}_{P-1}({\bf x}_{1}) \\
#         {\phi}_{0}({\bf x}_{2}) & {\phi}_{1}({\bf x}_{2}) & {\phi}_{2}({\bf x}_{2}) & \cdots & {\phi}_{P-1}({\bf x}_{2}) \\
#         \vdots & \vdots & \vdots & \ddots & \vdots \\
#         {\phi}_{0}({\bf x}_{N}) & {\phi}_{1}({\bf x}_{N}) & {\phi}_{2}({\bf x}_{N}) & \cdots & {\phi}_{P-1}({\bf x}_{N}) \\
#     \end{array}\right]
# \\
# {\bf X} = {[{\bf x}_{1}, {\bf x}_{2}, ..., {\bf x}_{N}]}^{T}
# $$
# 
# This consists of the following methods and fields:
# 
# <span style="color: red; ">**methods**</span>
# * generatePhi(self, type, order=None, sigma=None) :
# 
#     generate Phi($={\bf \Phi}$) from ${\bf X}$ given in constructor. The basis functions is determined by *type*('linear', 'polynomial' or 'gaussian'). `order`($=d_{p}$) is ignored if we don't choose the polynomial, and `sigma`($={\sigma}$) if we don't choose the Gaussian.
#     Actually This method simply calls the following three private methods.
# 
# * __generatePhiL(self) : *private*
# 
#     generate Phi($={\bf \Phi}$) from ${\bf X}$ given in constructor using <span style="color: red; ">**the Linear Function**</span>. Actually we only need to insert a column filled with 1 in 0th column.
# 
# $$
# {\phi}_{i}({\bf x}_{n}) = \left\{\begin{array}{ll}
#                                     1 & i = 0 \\
#                                     {[{\bf x}_{n}]}_{i} & {\rm Otherwise}
#                                  \end{array}
#                            \right.
# $$
# 
# * __generatePhiP(self, order) : *private*
# 
#     generate Phi($={\bf \Phi}$) from ${\bf X}$ given in constructor using <span style="color: red; ">**the Polynomial Function**</span>. In this case, `order`(=$d_{p}$) is a parameter controling the dimentionality of the function and denotes the sum of $i_{j}$. This includes two inner methods to generate multi-index $i = (i_{1}, ..., i_{d_{in}})$ and to compute ${\phi}_{i}({\bf x}_{n})$.
# $$
# d_{p} = {\sum}^{d_{in}}_{j} {i}_{j} \\
# {\phi}_{i}({\bf x}_{n}) = {\prod}^{d_{in}}_{j} {({[{\bf x}_{n}]}_{j})}^{{i}_{j}}
# $$
# 
# * __generatePhiG(self, sigma) : *private*
# 
#     generate Phi($={\bf \Phi}$) from ${\bf X}$ given in constructor using <span style="color: red; ">**the Gaussian Function**</span>. This also includes a inner method to compute ${\phi}_{i}({\bf x}_{n})$.
# 
# $$
# {\phi}_{i}({\bf x}_{n}) = \left\{\begin{array}{ll}
#                                     1 & i = 0 \\
#                                     e^{-{||{\bf x}_{n}-{\bf x}_{i}||}^{2}/(2{\sigma}^{2})} & {\rm Otherwise}
#                                  \end{array}
#                            \right.
# $$
# 
# <span style="color: red; ">**fields**</span>
# 
# * X($={\bf X}$) :
# 
#     raw input matrix

# In[48]:


class PhiGenerator():
    def __init__(self, X: np.ndarray):
        self.X = X
        return
    
    def generatePhi(self, type: str, order: int=None, sigma: float=None):
        if type == 'linear':
            return self.__generatePhiL()
        if type == 'polynomial':
            return self.__gereratePhiP(order)
        if type == 'gaussian':
            return self.__generatePhiG(sigma)
        
        print('ERROR: invalid `type`. `type` should be \'linear\', \'polynomial\' or \'gaussian\'. returns None')
        return None

    def __generatePhiL(self):
        return np.insert(self.X, 0, 1, axis=1)
    
    def __gereratePhiP(self, order: int):
        if order is None:
            print('ERROR: `order` is None. returns None')
            return None
        
        def prepareMultiIndices(size: int, order: int):
            buckets = [[] * size for i in range(order + 1)]

            for multi_index in itertools.product(range(order+1), repeat=size):
                s = sum(multi_index)
                if s > order: continue
                # evacate the　generated list temporarily
                buckets[s].append(multi_index)
            
            # sort it in a preferred order
            multi_indices = []
            for bucket in buckets:
                for multi_index in reversed(bucket):
                    multi_indices.append(multi_index)

            return multi_indices
        
        def calc_phi(x_n: np.ndarray, i: list):
            if len(i) != x_n.shape[0]:
                print('WARNING: length of multi-index i != lengh of x_n')
            phi = 1
            for j, i_j in enumerate(i):
                phi = phi * (x_n[j] ** i_j)

            return phi
        
        multi_indices = prepareMultiIndices(self.X.shape[1], order)

        Phi = np.zeros([self.X.shape[0], len(multi_indices)])

        for row in range(self.X.shape[0]):
            for column, i in enumerate(multi_indices):
                Phi[row][column] = calc_phi(self.X[row], i)

        return Phi
    
    def __generatePhiG(self, sigma: float):
        if sigma is None:
            print('ERROR: `sigma` is None. returns None')
            return None

        def calc_phi(x_n: np.ndarray, x_i: np.ndarray, sigma: float):
            residue = x_n - x_i
            exponent = -np.sum(residue ** 2) / (2 * (sigma ** 2))
            return math.exp(exponent)

        Phi = np.zeros([self.X.shape[0], self.X.shape[0] + 1])

        for row in range(self.X.shape[0]):
            Phi[row][0] = 1
            for column in range(self.X.shape[0]):
                Phi[row][column + 1] = calc_phi(self.X[row], self.X[column], sigma)
        
        return Phi


# #### <span style="color: yellow; ">Module Test</span>

# As a test, we generate a toy sample:
# $$
# {\bf X} = 
#     \left[\begin{array}{c}
#         0 & 1 & 2 \\
#         1 & 2 & 3 \\
#         \vdots & \vdots & \vdots \\
#         4 & 5 & 6 \\
#     \end{array}\right]
# $$

# In[49]:


if __name__ == '__main__':
    num_samples = 5
    X = []
    for row in range(num_samples):
        X.append(list(range(row, row+3)))
    
    X = np.array(X)
    print(X)

    generator = PhiGenerator(X)


# By the linear basis function, ${\bf \Phi}$ will be:
# $$
# {\bf \Phi} = 
#     \left[\begin{array}{c}
#         1 & 0 & 1 & 2 \\
#         1 & 1 & 2 & 3 \\
#         \vdots & \vdots & \vdots \\
#         1 & 4 & 5 & 6 \\
#     \end{array}\right]
# $$

# In[50]:


if __name__ == '__main__':
    Phi = generator.generatePhi('linear')
    print(Phi)


# By the polynomial basis function with $d_{p} = 2$, ${\bf \Phi}$ will be:
# $$
# {\bf \Phi} = 
#     \left[\begin{array}{c}
#         1 & 0 & 1 & 2 & 0 & 0 & 0 & 1 & 2 & 4 \\
#         1 & 1 & 2 & 3 & 1 & 2 & 3 & 4 & 6 & 9 \\
#         \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
#         1 & 4 & 5 & 6 & 16 & 20 & 24 & 25 & 30 & 36\\
#     \end{array}\right]
# $$
# because ${\bf \Phi}_{{1, 5}}=0=0\times0, {\bf \Phi}_{{1, 6}}=0=0\times1, {\bf \Phi}_{{1, 7}}=2=0\times2, {\bf \Phi}_{{1, 8}}=0=1\times1, {\bf \Phi}_{(1, 9)}=2=1\times2, {\bf \Phi}_{(1, 10)}=4=2\times2$...

# In[51]:


if __name__ == '__main__':
    Phi = generator.generatePhi('polynomial', order=2)
    print(Phi)


# By the gaussian basis function with $\sigma = 1$, ${\bf \Phi}$ will be:
# $$
# {\bf \Phi} = 
#     \left[\begin{array}{c}
#         1 & 1 & e^{-3/2} & e^{-6} & e^{-27/2} & e^{-24}\\
#         1 & e^{-3/2} & 1 & e^{-3/2} & e^{-6} & e^{-27/2}\\
#         \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
#         1 & e^{-24} & e^{-27/2} & e^{-6} & e^{-3/2} & 1\\
#     \end{array}\right]
# $$
# because in this case ${||{\bf x}_{n}-{\bf x}_{i}||}^{2} = 3{(n-x)}^{2}$

# In[52]:


if __name__ == '__main__':
    Phi = generator.generatePhi('gaussian', sigma=1)
    print(Phi)
    print('e^(-3/2) = ' + str(np.exp(-3/2)))
    print('e^(-6) = ' + str(np.exp(-6)))
    print('e^(-27/2) = ' + str(np.exp(-27/2)))
    print('e^(-24) = ' + str(np.exp(-24)))


# In[53]:


if 'get_ipython' in globals():
    import subprocess
    subprocess.run(['jupyter', 'nbconvert', '--to', 'python', '*.ipynb'])
    print('Saved as Phi_generator.py')

