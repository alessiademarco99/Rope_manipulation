import numpy as np  

class SphereConstraint:
    def __init__(self, center, r, system):
        self.center = center
        self.r = r
        self.system = system

    def evaluate_constraint(self, x, k):	
		#evaluate the state to see if the constraint is violated
        x=x.reshape(6*self.system.n_masses,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))        

        length = (x_next[0] - self.center[0])**2 + (x_next[1] - self.center[1])**2 + (x_next[2] - self.center[2])**2
        return -(self.r**2 - length)

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
        

class BoxConstraint:
    def __init__(self,ux,lx,uy,ly,uz,lz,system, horizon):
        self.ux=ux
        self.uy=uy
        self.uz=uz
        self.lx=lx
        self.ly=ly
        self.lz=lz
        self.system=system
        self.horizon=horizon
        
    def evaluate_constraint(self,x):
        x=x.reshape(6*self.system.n_masses,1)
        x_next = self.system.transition(x, np.zeros(self.system.control_size))
        p,v=np.vsplit(x_next,2)
        p=p.reshape(self.system.n_masses,3)
        v=v.reshape(self.system.n_masses,3)
        for i in range(self.system.n_masses):
            if p[i,0] >= self.lx and p[i,0] <=self.ux:
                if p[i,1] >= self.ly and p[i,1] <=self.uy:
                    if p[i,2] >= self.lz and p[i,2] <=self.uz:
                        return 1    
        return -1       
      
                        
                    
    def evaluate_constraint_J(self, x):
        x=x.reshape(6*self.system.n_masses,1)

        result = -np.ones(x.shape) * self.system.dt
        for i in range(self.system.n_masses*3):
            result[i]= -1
        return np.ravel(result)
        
        
  