import numpy as np  
#import jax.numpy as np
# import jax 
# import jax.numpy as np
# from jax import jit, vmap, grad, value_and_grad, jacfwd
# from jax import random
# from jax.ops import index, index_add, index_update
# from jax.lax import cond, scan
# from jax.experimental import optimizers
# from jax.config import config
# config.update("jax_debug_nans", True)
# jax.config.update('jax_enable_x64', True)
# from functools import partial

  

class SphereConstraint:
    def __init__(self, center, r, system):
        self.center = center
        self.r = r
        self.system = system
        #self.start=i
    
    #@partial(jit)
    def evaluate_constraint(self, x, k):	
		#evaluate the state to see if the constraint is violated
        x=x.reshape(6*self.system.n_masses,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))        
        #print(x_next) 
        length = (x_next[0] - self.center[0])**2 + (x_next[1] - self.center[1])**2 + (x_next[2] - self.center[2])**2
        return -(self.r**2 - length)

    
    #@jit
    def evaluate_constraint_J(self, x, k):
		#evolve the system for one to evaluate constraint
        x=x.reshape(6*self.system.n_masses,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))
        result = np.zeros(x.shape)
        result[0] = 2*(x_next[0] - self.center[0])
        result[1] = 2*(x_next[1] - self.center[1])
        result[2] = 2*(x_next[2] - self.center[2])
        result[self.system.n_masses] = 2*(x_next[0] - self.center[0]) * self.system.dt
        result[self.system.n_masses+1] = 2*(x_next[1] - self.center[1]) * self.system.dt
        result[self.system.n_masses+2] = 2*(x_next[2] - self.center[2]) * self.system.dt
        return np.ravel(result)
        
class CircleConstraint:
    def __init__(self, center, r, system, i):
        self.center = center
        self.r = r
        self.system = system
        self.start=i
    
    #@jit
    def evaluate_constraint(self, x, k):	
		#evaluate the state to see if the constraint is violated
        x=x.reshape(6*self.system.n_masses,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))        
        #print(x_next) 
        if k in [2990,2991,2992,2993,2994,2995,2996,2997,2998,2998,2999,3000]:
            length = (x_next[self.start] - self.center[0])**2 + (x_next[self.start+1] - self.center[1])**2
        #print(x_next[self.start:self.start+2], self.r**2 - length)
            return -(self.r**2 - length)
        else:
            return -1
    
    #@jit
    def evaluate_constraint_J(self, x, k):
		#evolve the system for one to evaluate constraint
        x=x.reshape(6*self.system.n_masses,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))
        result = np.zeros(x.shape)
        if k in [2990,2991,2992,2993,2994,2995,2996,2997,2998,2998,2999,3000]:
            result[self.start] = 2*(x_next[self.start] - self.center[0])
            result[self.start+1] = 2*(x_next[self.start+1] - self.center[1])
            result[self.start+self.system.n_masses] = 2*(x_next[self.start] - self.center[0]) * self.system.dt
            result[self.start+self.system.n_masses+1] = 2*(x_next[self.start+1] - self.center[1]) * self.system.dt
            return np.ravel(result)
        else:
            return np.ravel(result)
# class BoxSurface:
#     def __init__(self, center, r, system): 
#         self.center = center #array
#         self.r = r #array
#         self.system = system
        
#     def evaluate_constraint(self, x):
#         x_next = self.system.transition(x, np.zeros(self.system.control_size))
#         length = np.array([abs(x_next[0] - self.center[0]) , abs(x_next[1] - self.center[1]), abs(x_next[2] - self.center[2])])
#         return (self.r - length)

class BoxSurface_xy:
    def __init__(self,center,dist,system):
        self.center=center
        self.dist=dist
        self.system=system
        
    def evaluate_constraint(self,x):
        x=x.reshape(18,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))
        p,_=np.vsplit(x_next,2)
        p=p.reshape(3,3)
        px=p[2,0]
        distance=np.max(abs(px-self.center))
        return -(self.dist-distance) #should be inside
    
    def evaluate_constraint_J(self, x):
        x=x.reshape(18,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))
        result = np.zeros(x.shape)
        result[0] = -1
        result[1] = -1
        result[2] = -1
        result[3] = - self.system.dt
        result[4] = - self.system.dt
        result[5] = - self.system.dt
        return np.ravel(result)
        
  