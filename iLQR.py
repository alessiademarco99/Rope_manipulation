import numpy as np   
from numpy.linalg import inv

class LQR:
    def __init__(self, system, initial_state, horizon, constraints):
        self.system=system
        self.horizon=horizon
        self.initial_state = np.copy(initial_state)
        #self.best_J = 1000
        self.reg_factor_u = 0.001
        self.constraints=constraints[:]
        self.multipliers=np.zeros((len(self.constraints),self.horizon))
        self.mu=np.zeros((len(self.constraints),self.horizon))
       
        self.penalty=10
        self.alpha=1
        self.e_constraint=0.1
        self.d=np.zeros((self.system.control_size, self.horizon))
        self.K=np.zeros((self.system.control_size, self.system.state_size, self.horizon))
        self.delta_V1=np.zeros(self.horizon)
        self.delta_V2=np.zeros(self.horizon)
        self.max_iter=35
        
    def set_initial_trajectories(self, x_traj, u_traj):
        self.x_traj = np.copy(x_traj)
        self.u_traj = np.copy(u_traj)
        
    #provare a non modificare self.best_J ma argomenti e returns
    def forward_pass(self,x_up,u_up,J_prev):
        done=0
        new_J=0
        current_J=J_prev
        self.alpha=1
        x_new_traj = np.zeros((self.system.state_size, self.horizon + 1))
        u_new_traj = np.zeros((self.system.control_size, self.horizon))
        x = np.copy(self.initial_state)
        iter=0
        while done==0:
            for i in range(self.horizon):
                x_new_traj[:, i] = np.copy(np.ravel(x))
                
                delta_x = x - x_up[:, i].reshape(6*self.system.n_masses,1)
                u=u_up[:,i]+np.ravel(self.K[:,:,i].dot(delta_x))+self.alpha*self.d[:,i]
                u_new_traj[:, i] = np.copy(np.ravel(u))
                new_J += self.system.calculate_cost(x, u)
                x = self.system.transition(x, u)
                print("Forward pass")
            
            # print(new_J.shape)    
            x_new_traj[:, self.horizon] = np.copy(np.ravel(x))
            new_J += self.system.calculate_final_cost(x)
            
            adelta_V=self.alpha*self.delta_V1+self.alpha**2*self.delta_V2
            #J using x_up and u_up
            J=0
            for i in range(self.horizon):
                J+= self.system.calculate_cost(x_up[:,i], u_up[:,i])
            J+= self.system.calculate_final_cost(x_up[:,self.horizon])
            #z=(current_J-new_J)/(-np.sum(adelta_V)) #problema
            z=(J-new_J)/(-np.sum(adelta_V))
            print("z ",z)
            print("J_prev ",J)
            print("J_new ",new_J)
            iter+=1
            print("iters ",iter)
            if iter == self.max_iter:
                print('max iter')
                return x_up,u_up, J_prev
            if (z < 10 and z > 0.00001): #or iter==1: #line search condition
                x_up=np.copy(x_new_traj)
                u_up=np.copy(u_new_traj)
                
                print("forward ",new_J)
                done=1
                print("Forward pass required: ", iter, " iterations")
                iter=0
                return x_up,u_up,new_J
                
            else:
                new_J=0
                self.alpha=0.8*self.alpha
                x_new_traj = np.zeros((self.system.state_size, self.horizon + 1))
                u_new_traj = np.zeros((self.system.control_size, self.horizon))
                x = np.copy(self.initial_state)
        
        
    def backward_pass(self, lam, mu, x_new,u_new):
        print("Backward pass")
        self.mu=np.copy(mu)
        self.multipliers=np.copy(lam)
        
        # definition of pn and Pn
        ln_xx = np.copy(self.system.Q_f) 
        ln_x = self.system.Q_f @ (x_new[:, self.horizon] - self.system.goal)
        self.Iu=self.mu[:,self.horizon-1]*np.diag(np.ones(len(self.constraints)))
        C=np.ones(len(self.constraints))
        C_x=np.zeros((len(self.constraints),self.system.state_size))
        for i in range(len(self.constraints)):
            C[i]=self.constraints[i].evaluate_constraint(x_new[:, self.horizon], self.horizon)
            
        # checking if the constraint is active
            if C[i]  < - self.e_constraint and self.multipliers[i,self.horizon-1] == 0: 
                self.Iu[i,i] = 0 # it means that the constraint is not active
            
            C_x[i,:]=self.constraints[i].evaluate_constraint_J(x_new[:, self.horizon], self.horizon)
        cu=np.zeros((len(self.constraints),self.system.control_size)) # no constraints on control trajectory
        pn=ln_x+C_x.T @ (self.multipliers[:,self.horizon-1]+self.Iu @ C)
        Pn=ln_xx+C_x.T @ self.Iu @ C_x
        
        for i in range(self.horizon - 1, -1, -1):
            print(i)
            u = u_new[:, i]
            x = x_new[:, i]
            # definition of derivatives of the cost function
            l_xt = self.system.Q @ (x - self.system.goal)
            l_ut = self.system.R @ u
            l_uxt = np.zeros((self.system.control_size, self.system.state_size))
            l_xxt = np.copy(self.system.Q)
            l_uut = np.copy(self.system.R)
            self.Iu=self.mu[:,i]*np.diag(np.ones(len(self.constraints)))
            for j in range(len(self.constraints)):
                C_x[j,:]=self.constraints[j].evaluate_constraint_J(x,i)
                C[j]=self.constraints[j].evaluate_constraint(x,i)
                if C[j]  < - self.e_constraint and self.multipliers[j,i] == 0:
                    self.Iu[j,j] = 0
            if i+1==self.horizon:
                p=pn
                P=Pn
            else:
                p=Q_x + self.K[:,:,i+1].T @ Q_uu @ self.d[:,i+1] + self.K[:,:,i+1].T @ Q_u + Q_ux.T @ self.d[:,i+1]
                P=Q_xx + self.K[:,:,i+1].T @ Q_uu @ self.K[:,:,i+1]+ self.K[:,:,i+1].T @ Q_ux + Q_ux.T @ self.K[:,:,i+1]
            # print("I'm here") 
            A, B = self.system.transition_J(x,u)  #matrices A and B from dynamics
            Q_x = l_xt + A.T @ p + C_x.T @ (self.multipliers[:,i]+self.Iu @ C)
            Q_u = l_ut + B.T @ p + cu.T @ (self.multipliers[:,i]+self.Iu @ C)
            Q_ux = l_uxt + B.T @ P @ A + cu.T @ self.Iu @ C_x
            Q_uu = l_uut + B.T @ P @ B + cu.T @ self.Iu @ cu + self.reg_factor_u * np.identity(self.system.control_size)
            Q_xx = l_xxt + A.T @ P @ A + C_x.T @ self.Iu @ C_x
            # print(np.linalg.eigvals(Q_uu) > 0)
            if np.all(np.linalg.eigvals(Q_uu) > 0):
                
                self.K[:,:,i]=-inv(Q_uu) @ Q_ux
                self.d[:,i]=-inv(Q_uu) @ Q_u
                self.delta_V1[i]=self.d[:,i].T @ Q_u
                self.delta_V2[i]=0.5*self.d[:,i].T @ Q_uu @ self.d[:,i]
            else:
                self.reg_factor_u = self.reg_factor_u*5