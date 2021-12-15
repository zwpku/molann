#!/usr/bin/env python

# + 
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import torch
import math 
import torch.nn as nn
from sklearn.model_selection import train_test_split as ttsplit
# - 

### Part 1: define necessary functions and classes.

# The follow cell defines all our potentials classes.

# +
def g(a):
    """Gaussian function

    :param a: float, real value
    :return: float, g(a)
    """
    return np.exp(- a ** 2)

class TripleWellPotential:
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, *argv):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.dim = 2
        
    def V(self, X):
        """Potential fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: V: np.array, array of potential energy values
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        x = X[:,0]
        y = X[:,1]
        u = g(x) * (g(y - 1/3) - g(y - 5/3))
        v = g(y) * (g(x - 1) + g(x + 1))
        V = 3 * u - 5 * v + 0.2 * (x ** 4) + 0.2 * ((y - 1/3) ** 4)
        return V
    
    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """
        u = g(x) * (g(y - 1/3) - g(y - 5/3))
        a = g(y) * ((x - 1)*g(x - 1) + (x + 1) * g(x + 1))
        dVx = -6 * x * u + 10 * a + 0.8 * (x ** 3)
        return dVx
    
    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        u = g(x) * ((y - 1/3) * g(y - 1/3) - (y - 5/3) * g(y - 5/3))
        b = g(y) * (g(x - 1) + g(x + 1))
        dVy = -6 * u + 10 * y * b + 0.8 * ((y - 1/3) ** 3)
        return dVy
    
    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        return np.column_stack( (self.dV_x(X[:,0], X[:,1]), self.dV_y(X[:,0], X[:,1])) )
        
class TripleWellOneChannelPotential:
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, *argv):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.dim = 2
        
    def V(self, X):
        """Potential fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: V: np.array, array of potential energy values
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)   

        a = + 10 * np.exp(- 25 * X[:,0] ** 2 - (X[:,1] + (1 / 3)) ** 2)
        b = - 3 * np.exp(- X[:,0] ** 2 - (X[:,1] - (5 / 3)) ** 2)
        c = - 5 * np.exp(- X[:,1] ** 2 - (X[:,0] - 1) ** 2)
        d = - 5 * np.exp(- X[:,1] ** 2 - (X[:,0] + 1) ** 2) 
        e = + 0.2 * X[:,0] ** 4 + 0.2 * (X[:,1] - (1 / 3)) ** 4
        V = a + b + c + d + e 
        return V
    
    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """
        a = - 20 * 25 * x * np.exp(- 25 * x ** 2 - (y + (1 / 3)) ** 2)
        b = + 6 * x * np.exp(- x ** 2 - (y - (5 / 3)) ** 2)
        c = + 10 * (x - 1) * np.exp(- y ** 2 - (x - 1) ** 2)
        d = + 10 * (x + 1) * np.exp(- y ** 2 - (x + 1) ** 2) 
        e = + 0.8 * x ** 3
        dVx = a + b + c + d + e
        return dVx
    
    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        a = - 20 * (y + (1 / 3)) * np.exp(- 25 * x ** 2 - (y + (1 / 3)) ** 2)
        b = + 6 * (y - (5 / 3)) * np.exp(- x ** 2 - (y - (5 / 3)) ** 2)
        c = + 10 * y * np.exp(- y ** 2 - (x - 1) ** 2)
        d = + 10 * y * np.exp(- y ** 2 - (x + 1) ** 2) 
        e = + 0.8 * y ** 3
        dVy = a + b + c + d + e
        return dVy
    
    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        return np.column_stack( (self.dV_x(X[:,0], X[:,1]), self.dV_y(X[:,0], X[:,1])) )
        
class DoubleWellPotential:
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, *argv):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        """
        self.beta = beta
        self.dim = 2
        
    def V(self, X):
        """Potential fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: V: np.array, array of potential energy values
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        a = 3 * np.exp(- X[:,0] ** 2 - X[:,1] ** 2)
        b = - 5 * np.exp(- X[:,1] ** 2 - (X[:,0] - 1) ** 2)
        c = - 5 * np.exp(- X[:,1] ** 2 - (X[:,0] + 1) ** 2)
        d = + 0.2 * X[:,0] ** 4 + 0.2 * X[:,1] ** 4
        V = a + b + c + d
        return V
    
    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """   
        a = -6 * x * np.exp(- x ** 2 - y ** 2)
        b = + 10 * (x - 1) * np.exp(- y ** 2 - (x - 1) ** 2)
        d = + 10 * (x + 1) * np.exp(- y ** 2 - (x + 1) ** 2)
        c = + 0.8 * x ** 3
        dVx = a + b + c + d
        return dVx
    
    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """
        a = - 6 * y * np.exp(- x ** 2 - y ** 2) 
        b = + 10 * y * np.exp(- y ** 2 - (x - 1) ** 2)
        c = + 10 * y * np.exp(- y ** 2 - (x + 1) ** 2)
        d = + 0.8 * y ** 3
        dVy = a + b + c + d
        return dVy
    
    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        return np.column_stack( (self.dV_x(X[:,0], X[:,1]), self.dV_y(X[:,0], X[:,1])) )
        
class ZPotential:
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, *argv):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.dim = 2
        
    def V(self, X):
        """Potential fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: V: np.array, array of potential energy values
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        a = - 3 * np.exp(- 0.01 * (X[:,0] + 5) ** 2 - 0.2 * (X[:,1] + 5) ** 2) 
        b = - 3 * np.exp(- 0.01 * (X[:,0] - 5) ** 2 - 0.2 * (X[:,1] - 5) ** 2)
        c = + 5 * np.exp(- 0.20 * (X[:,0] + 3 * (X[:,1] - 3)) ** 2) / (1 + np.exp(- X[:,0] - 3))
        d = + 5 * np.exp(- 0.20 * (X[:,0] + 3 * (X[:,1] + 3)) ** 2) / (1 + np.exp(+ X[:,0] - 3))
        e = + 3 * np.exp(- 0.01 * (X[:,0] ** 2 + X[:,1] ** 2))
        f = (X[:,0] ** 4 + X[:,1] ** 4) / 20480
        V = a + b + c + d + e + f
        return V
    
    def dV_x(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVx: float, derivative of the potential with respect to x
        """    
        a = + 0.06 * (x + 5) * np.exp(- 0.01 * (x + 5) ** 2 - 0.2 * (y + 5) ** 2)
        b = + 0.06 * (x - 5) * np.exp(- 0.01 * (x - 5) ** 2 - 0.2 * (y - 5) ** 2)
        d = + (5 / (1 + np.exp(- x - 3)) ** 2) * (+ np.exp(- x - 3) * np.exp(- 0.2 * (x + 3 * (y - 3)) ** 2) - 0.4 * (x + 3 * (y - 3)) * np.exp(-0.2 * (x + 3 * (y - 3)) ** 2) * (1 + np.exp(- x - 3)))
        c = + (5 / (1 + np.exp(+ x - 3)) ** 2) * (- np.exp(+ x - 3) * np.exp(- 0.2 * (x + 3 * (y + 3)) ** 2) - 0.4 * (x + 3 * (y + 3)) * np.exp(-0.2 * (x + 3 * (y + 3)) ** 2) * (1 + np.exp(+ x - 3)))
        e = - 0.06 * x * np.exp(- 0.01 * (x ** 2 + y ** 2))
        f = (4 * x ** 3) / 20480
        dVx = a + b + c + d + e + f  
        return dVx
    
    def dV_y(self, x, y):
        """
        :param x: float, x coordinate
        :param y: float, y coordinate

        :return: dVy: float, derivative of the potential with respect to y
        """   
        a = + 1.2 * (y + 5) * np.exp(- 0.01 * (x + 5) ** 2 - 0.2 * (y + 5) ** 2)
        b = + 1.2 * (y - 5) * np.exp(- 0.01 * (x - 5) ** 2 - 0.2 * (y - 5) ** 2)
        c = - (5 / (1 + np.exp(- x - 3))) * 1.2 * (x + 3 * (y - 3)) * np.exp(- 0.2 * (x + 3 * (y - 3)) ** 2)
        d = - (5 / (1 + np.exp(+ x - 3))) * 1.2 * (x + 3 * (y + 3)) * np.exp(- 0.2 * (x + 3 * (y + 3)) ** 2)
        e = -  0.06 * y * np.exp(- 0.01 *(x ** 2 + y ** 2))
        f = (4 * y ** 3) / 20480
        dVy = a + b + c + d + e + f
        return dVy
    
    def nabla_V(self, X):
        """Gradient of the potential energy fuction

        :param X: np.array, array of position vectors (x,y), ndim = 2, shape = (,2)
        :return: grad(X): np.array, array of gradients with respect to position vector (x,y), ndim = 2, shape = (,2)
        """
        assert(type(X) == np.ndarray)
        assert(X.ndim == 2)
        assert(X.shape[1] == 2)
        return np.column_stack( (self.dV_x(X[:,0], X[:,1]), self.dV_y(X[:,0], X[:,1])) )
        
class TripleWellPotAlongCircle :
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, eps):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.eps = eps 
        self.dim = 2

    def V(self, x):
      # angle in [-pi, pi] 

      theta = np.arctan2(x[:,1], x[:,0])
      # radius
      r = np.sqrt( x[:,0] * x[:,0] + x[:,1] * x[:,1] )

      v_vec = np.zeros(len(x))
      for idx in range(len(x)) :
          # potential V_1
          if theta[idx] > math.pi / 3 : 
            v_vec[idx] = (1-(theta[idx] * 3 / math.pi- 1.0)**2)**2
          if theta[idx] < - math.pi / 3 : 
            v_vec[idx] = (1-(theta[idx] * 3 / math.pi + 1.0)**2)**2
          if theta[idx] > - math.pi / 3 and theta[idx] < math.pi / 3:
            v_vec[idx] = 3.0 / 5.0 - 2.0 / 5.0 * np.cos(3 * theta[idx])  
      # potential V_2
      v_vec = v_vec * 1.0 + (r - 1)**2 * 1.0 / self.eps + 5.0 * np.exp(-5.0 * r**2) 
      return v_vec

    def nabla_V(self, x): 
      # angle
      theta = np.arctan2(x[:,1], x[:,0])
      # radius
      r = np.sqrt( x[:,0] * x[:,0] + x[:,1] * x[:,1] )

      if any(np.fabs(r) < 1e-8): 
          print ("warning: radius is too small! r=%.4e" % r)
      dv1_dangle = np.zeros(len(x))
      # derivative of V_1 w.r.t. angle
      for idx in range(len(x)) :
          if theta[idx] > math.pi / 3: 
            dv1_dangle[idx] = 12 / math.pi * (theta[idx] * 3 / math.pi - 1) * ((theta[idx] * 3 / math.pi- 1.0)**2-1)
          if theta[idx] < - math.pi / 3: 
            dv1_dangle[idx] = 12 / math.pi * (theta[idx] * 3 / math.pi + 1) * ((theta[idx] * 3 / math.pi + 1.0)**2-1)
          if theta[idx] > -math.pi / 3 and theta[idx] < math.pi / 3:
            dv1_dangle[idx] = 1.2 * math.sin (3 * theta[idx])
      # derivative of V_2 w.r.t. angle
      dv2_dangle = np.zeros(len(x))
      # derivative of V_2 w.r.t. radius
      dv2_dr = 2.0 * (r-1.0) / self.eps - 50.0 * r * np.exp(-r**2/0.2)

      return np.column_stack((-(dv1_dangle + dv2_dangle) * x[:,1] / (r * r)+ dv2_dr * x[:,0] / r,  (dv1_dangle + dv2_dangle) * x[:,0] / (r * r)+ dv2_dr * x[:,1] / r))

class StiffPot : 
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, eps):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.eps = eps 
        self.dim = 2

    def V(self, x):
      return (x[:,0]**2 - 1)**2 + 1.0 / self.eps * (x[:,0]**2 + x[:,1] - 1)**2

    def nabla_V(self, x): 
      return np.column_stack(( 4.0 * x[:,0] * (x[:,0]**2 - 1.0 + 1.0 / self.eps * (x[:,0]**2 + x[:,1] - 1)), 2.0 / self.eps * (x[:,0]**2 + x[:,1] - 1)) )

class UniformPotAlongCircle :
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, eps):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.eps = eps 
        self.dim = 2

    def V(self, x):
      return 1.0 / self.eps * (x[:,0]**2 + x[:,1]**2 - 1)**2

    def nabla_V(self, x): 
      return np.column_stack( (4.0 * x[:,0] / self.eps * (x[:,0]**2 + x[:,1]**2 - 1), 4.0 * x[:,1] / self.eps * (x[:,0]**2 + x[:,1]**2 - 1)) )

class DoubleWellPotAlongCircle :
    """Class to gather methods related to the potential function"""
    def __init__(self, beta, eps):
        """Initialise potential function class

        :param beta: float,  inverse temperature = 1 / (k_B * T)
        :param Z: float, partition function (computed below)
        """
        self.beta = beta
        self.eps = eps 
        self.dim = 2

    def V(self, x):
        return 2.0 * x[:,1]**2 + 1.0 / self.eps * (x[:,0]**2 + x[:,1]**2 - 1)**2

    def nabla_V(self, x): 
        return np.column_stack( (4.0 * x[:,0] / self.eps * (x[:,0]**2 + x[:,1]**2 - 1), 4.0 * x[:,1] + 4.0 * x[:,1] / self.eps * (x[:,0]**2 + x[:,1]**2 - 1)) )

# - 

# We then define a function 'UnbiasedTraj' to generate an trajectory according an Euler--Maruyama discretization 
# $$
# X^{n+1} = X^n - \Delta t \nabla V(X^n) + \sqrt{\frac{2 \Delta t}{\beta}} \, G^n 
# $$
# of the overdamped Langevin dynamics
# $$
# dX_t = -\nabla V(X_t) \, dt + \sqrt{\frac{2}{\beta}} \, dW_t
# $$
# This functions takes as argument a potential object, initial conditions, the number of simulation steps and a time step. It generates a realization of a trajectory (subsampled at some prescribed rate), and possibly records the value of the potential energy function at the points along the trajectory.

# +
def UnbiasedTraj(pot, X_0, delta_t=1e-3, T=1000, save=1, save_energy=False, seed=0):
    """Simulates an overdamped langevin trajectory with a Euler-Maruyama numerical scheme 

    :param pot: potential object, must have methods for energy gradient and energy evaluation
    :param X_0: Initial position, must be a 2D vector
    :param delta_t: Discretization time step
    :param T: Number of points in the trajectory (the total simulation time is therefore T * delta_t)
    :param save: Integer giving the period (counted in number of steps) at which the trajectory is saved
    :param save_energy: Boolean parameter to save energy along the trajectory

    :return: traj: np.array with ndim = 2 and shape = (T // save + 1, 2)
    :return: Pot_values: np.array with ndim = 2 and shape = (T // save + 1, 1)
    """
    r = np.random.RandomState(seed)
    X = X_0.reshape(1,2)
    dim = X.shape[1]
    traj = [X[0,:]]
    if save_energy:
        Pot_values = [pot.V(X)]
    else:
        Pot_values = None
    for i in range(T):
        b = r.normal(size=(dim,))
        X = X - pot.nabla_V(X.reshape(1,2)) * delta_t + np.sqrt(2 * delta_t/pot.beta) * b
        if i % save==0:
            traj.append(X[0,:])
            if save_energy:
                Pot_values.append(pot.V(X)[0])
    return np.array(traj), np.array(Pot_values)

# -

# We now define the Auto encoders classes and useful functions for the training.

# +
class DeepAutoEncoder(nn.Module):
    def __init__(self, encoder_dims, decoder_dims):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param encoder_dims: list, List of dimensions for encoder, including input/output layers
        :param decoder_dims: list, List of dimensions for decoder, including input/output layers
        """
        super(DeepAutoEncoder, self).__init__()
        layers = []
        for i in range(len(encoder_dims)-2) :
            layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i+1])) 
            layers.append( torch.nn.Tanh() )
        layers.append(torch.nn.Linear(encoder_dims[-2], encoder_dims[-1])) 

        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims)-2) :
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i+1])) 
            layers.append( torch.nn.Tanh() )
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1])) 

        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, inp):
        # Input Linear function
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded

def set_learning_parameters(model, learning_rate, loss='MSE', optimizer='Adam'):
    """Function to set learning parameter

    :param model: Neural network model build with PyTorch,
    :param learning_rate: Value of the learning rate
    :param loss: String, type of loss desired ('MSE' by default, another choice leads to L1 loss)
    :param optimizer: String, type of optimizer ('Adam' by default, another choice leads to SGD)

    :return:
    """
    #--- chosen loss function ---
    if loss == 'MSE':
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.L1Loss()
    #--- chosen optimizer ---
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    return loss_function, optimizer

def xi_ae(model,  x):
    """Collective variable defined through an auto encoder model

    :param model: Neural network model build with PyTorch
    :param x: np.array, position, ndim = 2, shape = (1,1)

    :return: xi: np.array
    """
    model.eval()
    if torch.is_tensor(x) == False :
        x = torch.from_numpy(x).float()
    return model.encoder(x).detach().numpy()

def grad_xi_ae(model, x):
    """Gradient of the collective variable defined through an auto encoder model

    :param model: Neural network model build with pytorch,
    :param x: np.array, position, ndim = 2, shape = (1,1)

    :return: grad_xi: np.array
    """
    model.eval()
    if torch.is_tensor(x) == False :
        x = torch.tensor(x.astype('float32'))
    x.requires_grad_()
    enc = model.encoder(x)
    grad = torch.autograd.grad(enc, x)[0]
    return grad.detach().numpy()


# Next, we define the training function to compute and store values for the "variance interpretation" of the training. 

def train_with_variance_decomposition_plots(model,
                                            loss_function,
                                            optimizer,
                                            traj,
                                            weights,
                                            num_epochs=10,
                                            batch_size=32,
                                            test_size=0.2,
                                            n_bins_z=20,
                                            x_domain=[-2, 2],
                                            y_domain=[-1.5, 2.5]
                                           ):
    """Function to train an AE model
    
    :param model: Neural network model built with PyTorch,
    :param loss_function: Function built with PyTorch tensors or built-in PyTorch loss function
    :param optimizer: PyTorch optimizer object
    :param traj: np.array, physical trajectory (in the potential pot), ndim == 2, shape == T // save + 1, pot.dim
    :param weights: np.array, weights of each point of the trajectory when the dynamics is biased, ndim == 1, shape == T // save + 1, 1
    :param num_epochs: int, number of times the training goes through the whole dataset
    :param batch_size: int, number of data points per batch for estimation of the gradient
    :param test_size: float, between 0 and 1, giving the proportion of points used to compute test loss
    :param n_bins_z: integer, number of bins in the z coordinat
    :param x_domain: list, min and max value of x to define the interval of variation of the encoded values
    :param y_domain: list, min and max value of y to define the interval of variation of the encoded values

    :return: model, trained neural net model
    :return: loss_list, list of lists of train losses and test losses; one per batch per epoch
    :return: X_given_z, list giving, for each epoch, a list where, for each bin in z, a list of X vectors is provided 
    :return: z_bins, list giving, for each epoch, the list of bins centers in z (correspond to a grid)
    """
    #--- prepare the data ---
    # split the dataset into a training set (and its associated weights) and a test set
    X_train, X_test, w_train, w_test = ttsplit(traj, weights, test_size=test_size)
    X_train = torch.tensor(X_train.astype('float32'))
    X_test = torch.tensor(X_test.astype('float32'))
    w_train = torch.tensor(w_train.astype('float32'))
    w_test = torch.tensor(w_test.astype('float32'))
    # intialization of the methods to sample with replacement from the data points (needed since weights are present)
    train_sampler = torch.utils.data.WeightedRandomSampler(w_train, len(w_train))
    test_sampler  = torch.utils.data.WeightedRandomSampler(w_test, len(w_test))
    # method to construct data batches and iterate over them
    train_loader = torch.utils.data.DataLoader(dataset=X_train,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=train_sampler)
    test_loader  = torch.utils.data.DataLoader(dataset=X_test,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               sampler=test_sampler)
    
    #--- Prepare empty list to store X given z ---
    X_given_z          = [[[] for i in range(n_bins_z)] for j in range(num_epochs)]
    z_bins = [[] for j in range(num_epochs)]
    # --- start the training over the required number of epochs ---
    loss_list = []
    print ("\ntraining starts, %d epochs in total." % num_epochs) 
    for epoch in range(num_epochs):
        # Train the model by going through the whole dataset
        model.train()
        train_loss = []
        for iteration, X in enumerate(train_loader):
            # Set gradient calculation capabilities
            X.requires_grad_()
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # Forward pass to get output
            out = model(X)
            # Evaluate loss
            loss = loss_function(out, X)
            # Get gradient with respect to parameters of the model
            loss.backward()
            # Store loss
            train_loss.append(loss)
            # Updating parameters
            optimizer.step()
        # Evaluate the test loss on the test dataset and preparation to compute conditional properties
        model.eval()

        with torch.no_grad():
            # Evaluation of test loss
            test_loss = []
            for iteration, X in enumerate(test_loader):
                out = model(X)
                # Evaluate loss
                loss = loss_function(out, X)
                # Store loss
                test_loss.append(loss)

            loss_list.append([torch.tensor(train_loss), torch.tensor(test_loss)])

            # Preparation for computation of conditional properties : first bounds on z, then sorting out

            xi_ae1_on_grid = xi_ae(model, X_train)[:,0]
            # equal-width bins
            z_bin = np.linspace(xi_ae1_on_grid.min(), xi_ae1_on_grid.max(), n_bins_z)
            # compute index of bin
            inds = np.digitize(xi_ae1_on_grid, z_bin) 

            # distribute train data to each bin
            for bin_idx in range(n_bins_z) : 
                X_given_z[epoch][bin_idx] = X_train[inds == bin_idx+1, :].detach()

        #--- Computation of conditional properties ---
        z_bins[epoch]= z_bin.tolist()

    print ("training ends.\n") 

    return model, loss_list, X_given_z, z_bins 

# -

### Part 2: run a test.

# All the parameters are set in the cell below. 

# +
pot_list = [ TripleWellPotential, TripleWellOneChannelPotential, DoubleWellPotential, ZPotential, \
                TripleWellPotAlongCircle, StiffPot, UniformPotAlongCircle, DoubleWellPotAlongCircle ] 
# choose a potential in the pot_list above
pot_id = 4

beta = 1.0
eps = 0.1

# for data generation
delta_t = 0.001
T = 100000
save = 10
seed = None 

# for training
batch_size = 10000
num_epochs = 500
learning_rate = 0.005
n_bins_z = 20          # number of bins in the Encoded dimension
optimizer_algo='Adam'  # Adam by default, otherwise SGD
#ae1 = SimpleAutoEncoder(2,1) 
ae1 = DeepAutoEncoder([2, 20, 20, 1], [1, 20, 20, 2]) 
print("test using NN:", ae1) 

save_fig_to_file = False

# -

# First, we visualise our different potentials.

# +
# x and y domains for each potential
x_domains = [[-2.5, 2.5], [-2.5, 2.5], [-2.5, 2.5], [-3.5, 3.5], [-2.5, 2.5], [-2, 2], [-1.5, 1.5], [-1.5, 1.5] ]
y_domains = [[-1.5, 2.5], [-1.5, 2.5], [-1.5, 2.5], [-3.5, 3.5], [-2.5, 2.5], [-2.0, 1.5], [-1.5, 1.5], [-1.5, 1.5] ]
x0_list = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1] ]
v_min_max = [[-4,3], [-4,3], [-4,7], [0.3,8], [0,5], [0,5], [0,5], [0,5] ]

pot = pot_list[pot_id](beta, eps)
x_domain = x_domains[pot_id] 
y_domain = y_domains[pot_id] 

pot_name = type(pot).__name__

print ('potential name: %s' % pot_name) 

gridx = np.linspace(x_domain[0], x_domain[1], 100)
gridy = np.linspace(y_domain[0], y_domain[1], 100)
x_plot = np.outer(gridx, np.ones(100)) 
y_plot = np.outer(gridy, np.ones(100)).T 

x2d = np.concatenate((x_plot.reshape(100 * 100, 1), y_plot.reshape(100 * 100, 1)), axis=1)

pot_on_grid = pot.V(x2d).reshape(100,100)
print ( "min and max values of potential: (%.4f, %.4f)" % (pot_on_grid.min(), pot_on_grid.max()) )

fig = plt.figure(figsize=(9,3))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2)
ax0.set_title(pot_name)
ax0.plot_surface(x_plot, y_plot, pot_on_grid , cmap='coolwarm', edgecolor='none')
ax1.pcolormesh(x_plot, y_plot, pot_on_grid, cmap='coolwarm',shading='auto', vmin=v_min_max[pot_id][0], vmax=v_min_max[pot_id][1])

if save_fig_to_file :
    filename = "%s.jpg" % pot_name
    fig.savefig(filename)
    print ( "potential profiles saved to file: %s" % filename )

# - 

# Data set generation 

# In the following we generate a trajectory in the desired potential with the function previously defined. 

# +
x_0 = np.array(x0_list[pot_id])

### Generate the trajectory
trajectory, _ = UnbiasedTraj(pot, x_0, delta_t=delta_t, T=T, save=save, save_energy=False, seed=seed)

### Plot the trajectory 
fig = plt.figure(figsize=(9,3))
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)
ax0.pcolormesh(x_plot, y_plot, pot_on_grid, cmap='coolwarm_r', shading='auto')
ax0.scatter(trajectory[:,0], trajectory[:,1], marker='x')
ax1.plot(range(len(trajectory[:,0])), trajectory[:,0], label='x coodinate along trajectory')

if save_fig_to_file :
    traj_filename = "traj_%s.jpg" % pot_name
    fig.savefig(traj_filename)
    print ("trajectory plot saved to file: %s" % traj_filename)

# - 

# Training the NN

# +
loss_function, optimizer = set_learning_parameters(ae1, learning_rate=learning_rate)

(
    ae1,
    loss_list ,
    X_given_z,
    z_bins
) = train_with_variance_decomposition_plots(ae1,
                                            loss_function,
                                            optimizer,
                                            trajectory,
                                            np.ones(trajectory.shape[0]),
                                            batch_size=batch_size,
                                            num_epochs=num_epochs,
                                            n_bins_z=n_bins_z,
                                            x_domain=x_domain,
                                            y_domain=y_domain
                              )

#--- Compute average train and test losses per epoch ---
loss_evol1 = []
for i in range(len(loss_list)):
    loss_evol1.append([torch.mean(loss_list[i][0]), torch.mean(loss_list[i][1])])
loss_evol1 = np.array(loss_evol1)

#--- Compute things to do "nice" plots ---

Esp_X_given_z   = [[np.zeros(2) for i in range(n_bins_z)] for j in range(num_epochs)]
Std1_X_given_z  = [[np.zeros(1) for i in range(n_bins_z)] for j in range(num_epochs)]
Std2_X_given_z  = [[np.zeros(2) for i in range(n_bins_z)] for j in range(num_epochs)]
Var_Esp_X_given_z = np.zeros(num_epochs)

f_dec_z = [[np.zeros(2) for i in range(n_bins_z)] for j in range(num_epochs)]

for epoch in range(num_epochs):
    for j in range(n_bins_z):       
        # test whether there are elements X in the j-th bin, and if yes, compute conditional properties 
        if len(X_given_z[epoch][j]) > 0:
            # mean value of coordinates given z
            Esp_X_given_z[epoch][j] = X_given_z[epoch][j].mean(dim=0)

            f_dec_z[epoch][j] = ae1(Esp_X_given_z[epoch][j]).detach().numpy()
            # compute the standard deviation in x and y coordinates # [set dim=0 to loop over points X]
            Std2_X_given_z[epoch][j] = X_given_z[epoch][j].std(dim=0).numpy()
            # variances taking both coordinates into account 
            Std1_X_given_z[epoch][j] = X_given_z[epoch][j].std().numpy()

            Esp_X_given_z[epoch][j] = Esp_X_given_z[epoch][j].detach().numpy()

    Var_Esp_X_given_z[epoch] = np.std(Esp_X_given_z[epoch])

# Obtain values of the RC and potential on a grid to plot contour lines
xi_ae1_on_grid = xi_ae(ae1, x2d).reshape(100, 100)

# -

# Plot the results 

# +
start_epoch_index = 1
fig, (ax0, ax1, ax2)  = plt.subplots(1,3, figsize=(12,4)) 
ax0.plot(range(start_epoch_index, num_epochs), loss_evol1[start_epoch_index:, 0], '--', label='train loss', marker='o')
ax0.plot(range(1, num_epochs), loss_evol1[start_epoch_index:, 1], '-.', label='test loss', marker='+')
ax0.legend()
ax0.set_title('losses')
ax1.pcolormesh(x_plot, y_plot, pot_on_grid, cmap='coolwarm_r',shading='auto')
ax1.set_title('potential: %s' % pot_name)
ax2.pcolormesh(x_plot, y_plot, xi_ae1_on_grid, cmap='coolwarm_r',shading='auto')
ax2.contour(x_plot, y_plot, xi_ae1_on_grid, 20, cmap = 'viridis')
ax2.set_title('RC')

if save_fig_to_file :
    fig_filename = 'training_loss_%s.jpg' % pot_name
    fig.savefig(fig_filename)
    print ('training loss plotted to file: %s' % fig_filename)

#--- variance of conditional averages ---
plt.figure()
plt.plot(range(num_epochs), Var_Esp_X_given_z)
plt.title('Variance of conditional averages as a function of epochs')
plt.xlabel('epoch number')

if save_fig_to_file :
    fig_filename = 'variance_%s.jpg' % pot_name
    plt.savefig(fig_filename)
    print ('variance plotted to file: %s' % fig_filename)

#--- Conditionnal expectancy and decoded values ---
index = -1
active_ind = [ len(X_given_z[index][j]) > 0 for j in range(n_bins_z)]
plt.figure()
plt.title('End of last epoch')
plt.plot(np.array(Esp_X_given_z[index])[active_ind, 0], np.array(Esp_X_given_z[index])[active_ind, 1], '-o', color='b', label='cond. avg.')
plt.plot(np.array(f_dec_z[index])[active_ind, 0], np.array(f_dec_z[index])[active_ind, 1], '*', color='black', label='decoder last epoch')
plt.pcolormesh(x_plot, y_plot, pot_on_grid, cmap='coolwarm_r',shading='auto')
index = 0
active_ind = [ len(X_given_z[index][j]) > 0 for j in range(n_bins_z)]
plt.plot(np.array(f_dec_z[index])[active_ind, 0], np.array(f_dec_z[index])[active_ind, 1], '*', color='pink', label='decoder first epoch')
plt.pcolormesh(x_plot, y_plot, pot_on_grid, cmap='coolwarm_r',shading='auto')
plt.legend()

if save_fig_to_file :
    fig_filename = 'result_%s.jpg' % pot_name
    plt.savefig(fig_filename)
    print ('conditional expectation plotted to file: %s' % fig_filename)
# -
