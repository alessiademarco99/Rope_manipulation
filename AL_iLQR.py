from iLQR import LQR
import numpy as np   

class AL_iLQR:
    def __init__(self, system, initial_state, horizon):
        self.system=system
        self.horizon=horizon
        self.x_traj=initial_state @ np.ones((1,self.horizon + 1))
        self.u_traj=0.001*np.ones((self.system.control_size, self.horizon))
        self.initial_state=np.copy(initial_state)
        self.constraints = []
        self.tol = 0.01
        self.tol_c = 0.5
        self.penalty=10 #phi
         
        self.max_iterations=100
        self.max_iterations_al=15
        
    def add_constraint(self, constraint):
        self.constraints.append(constraint)
        
    def algorithm(self):
        # Initialization of lambda, mu and phi
        self.multipliers=np.zeros((len(self.constraints),self.horizon))
        self.mu=np.ones((len(self.constraints),self.horizon))
        
        iter_al=0
        
        # initial value of the constraints
        c=np.ones((len(self.constraints),self.horizon))
        for i in range(len(self.constraints)): # for each constraint
            c[i,0]=self.constraints[i].evaluate_constraint(self.initial_state,0) 
        
        lqr=LQR(self.system, self.initial_state, self.horizon, self.constraints)
         
        while np.max(c) > self.tol_c and iter_al<=self.max_iterations_al:
            #minimize Lagrangian using iLQR
            iter_lq=0
            J_new=0
            J_prev=0
            # compute J with X and U
            for i in range(self.horizon):
                J_new += self.system.calculate_cost(self.x_traj[:,i],self.u_traj[:,i])
            J_new += self.system.calculate_final_cost(self.x_traj[:,self.horizon])
            
            x_new=np.copy(self.x_traj)
            u_new=np.copy(self.u_traj)
            self.diff=[]
            # line search
            while abs(J_new-J_prev) > self.tol and iter_lq<=self.max_iterations:  
                J_prev=J_new
                lqr.backward_pass(self.multipliers,self.mu,x_new,u_new)
                x_new,u_new,J_new=lqr.forward_pass(x_new,u_new,J_prev)
                iter_lq +=1
                print("diff J new e J prev ",abs(J_new-J_prev))
                self.diff.append(abs(J_new-J_prev))
                print("iter iLQR: ",iter_lq)
            print("Number of required iterations for iLQR: ",iter_lq)
            
            
            self.x_traj=np.copy(x_new)
            self.u_traj=np.copy(u_new)
            
            #update value for multipliers
            for i in range(len(self.constraints)): # for each constraint
                for k in range(self.horizon): 
                    
                    c[i,k]=self.constraints[i].evaluate_constraint(x_new[:,k],k) # constraint i at step k
                    
                    self.multipliers[i,k]=max(0, self.multipliers[i,k] + self.mu[i,k]*c[i,k])
                    self.mu[i,k]=self.penalty*self.mu[i,k]
            print(self.multipliers)
            iter_al=iter_al+1
            print("c", c)
            print("max c ",np.max(c))
            
            self.system.animate_cloth(self.horizon,self.x_traj,self.system.dt,F_history=self.u_traj, gifname=str(iter_al-1)+"_"+str(self.system.dt)+"_"+str(J_new)+"_"+str(np.max(c)))
            print(np.max(self.u_traj))
            print(np.min(self.u_traj))
        print("max c ",np.max(c))
        print("Number of required iterations for AL-iLQR: ",iter_al)
    
        