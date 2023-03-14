from iLQR import LQR
import numpy as np   

import matplotlib.pyplot as plt 

class AL_iLQR:
    def __init__(self, system, initial_state, horizon):
        self.system=system
        self.horizon=horizon
        self.x_traj=initial_state @ np.ones((1,self.horizon + 1))
        self.u_traj=0.001*np.ones((self.system.control_size, self.horizon))
        self.initial_state=np.copy(initial_state)
        self.constraints = []
        self.tol = 0.01
        self.penalty=8 #phi
         
        self.max_iterations=50
        self.max_iterations_al=50
        
    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        self.multipliers=np.zeros((len(self.constraints),self.horizon))
        self.mu=1e-4*np.ones((len(self.constraints),self.horizon))
        
    def algorithm(self):
        iter_al=0
        print("shape x_traje", self.x_traj.shape)
        print(self.x_traj)
        c=np.ones((len(self.constraints),self.horizon))
        for i in range(len(self.constraints)): # for each constraint
            c[i,0]=self.constraints[i].evaluate_constraint(self.initial_state,0)
        lqr=LQR(self.system, self.initial_state, self.horizon, self.constraints)
        
        # I'm checking on the constraint on all the instants, while I need to check only for the final state horizon N 
        # at least for the circle 
        while np.max(c) > self.tol and iter_al<=self.max_iterations_al: #it remains stuck
            #minimize Lagrangian
            iter_lq=0
            J_new=0
            J_prev=0
            # compute J with X and U
            for i in range(self.horizon):
                J_new += self.system.calculate_cost(self.x_traj[:,i],self.u_traj[:,i])
            J_new += self.system.calculate_final_cost(self.x_traj[:,self.horizon])
            print((self.x_traj[:,self.horizon]).shape)
            x_new=np.copy(self.x_traj)
            u_new=np.copy(self.u_traj)
            self.diff=[]
            while abs(J_new-J_prev) > self.tol and iter_lq<=self.max_iterations:  #non entra in alcune configurazioni
                J_prev=J_new
                lqr.backward_pass(self.multipliers,self.mu,x_new,u_new)
                x_new,u_new,J_new=lqr.forward_pass(x_new,u_new,J_prev)
                iter_lq +=1
                print("diff J new e J prev ",abs(J_new-J_prev))
                self.diff.append(abs(J_new-J_prev))
                print("iter iLQR: ",iter_lq)
            print("Number of required iterations for iLQR: ",iter_lq)
            plt.figure(iter_al)
            plt.scatter(np.arange(iter_lq),self.diff)
            plt.grid()
            print(self.diff)
            print(np.arange(iter_lq))
            self.x_traj=np.copy(x_new)
            self.u_traj=np.copy(u_new)
            print(x_new[12:15,:])
            #return
            #update value for multipliers
            for i in range(len(self.constraints)): # for each constraint
                for k in range(self.horizon): 
                    
                    c[i,k]=self.constraints[i].evaluate_constraint(x_new[:,k],k) # constraint i at step k
                    
                    self.multipliers[i,k]=max(0, self.multipliers[i,k] + self.mu[i,k]*c[i,k])
                    self.mu[i,k]=self.penalty*self.mu[i,k]
            iter_al=iter_al+1
            print("c", c)
            print("max c ",np.max(c))
            
            self.system.animate_cloth(self.x_traj, 0.001, gifname=str(iter_al-1))
        print("max c ",np.max(c))
        print("Number of required iterations for AL-iLQR: ",iter_al)
    
        