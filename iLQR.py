import numpy as np   
from numpy.linalg import inv
import shelve

class LQR:
    def __init__(self, system, initial_state, horizon, constraints):
        self.system=system
        self.horizon=horizon
        self.initial_state = np.copy(initial_state)
        self.reg_factor_u = 1e-3
        self.constraints=constraints[:]
        self.multipliers=np.zeros((len(self.constraints),self.horizon))
        self.mu=np.zeros((len(self.constraints),self.horizon))
       
        self.penalty=10
        self.alpha=1
        self.e_constraint=0.5
        self.d=np.zeros((self.system.control_size, self.horizon))
        self.K=np.zeros((self.system.control_size, self.system.state_size, self.horizon))
        self.delta_V1=np.zeros(self.horizon)
        self.delta_V2=np.zeros(self.horizon)
        self.max_iter=100
        self.fs=10
        
    def set_initial_trajectories(self, x_traj, u_traj):
        self.x_traj = np.copy(x_traj)
        self.u_traj = np.copy(u_traj)
        
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
            for i in range(0,self.horizon,self.fs):
                x_new_traj[:, i] = np.copy(np.ravel(x))
                delta_x = x - x_up[:, i].reshape(6*self.system.n_masses,1)
                u=u_up[:,i]+np.ravel(self.K[:,:,i].dot(delta_x))+self.alpha*self.d[:,i]
                # +
                u_new_traj[:, i] = np.copy(np.ravel(u))
                new_J += self.system.calculate_cost(x, u)
                x = self.system.transition(x, u)
                for j in range(1,self.fs):    # run the simulation for 10 times
                    u_new_traj[:, i+j] = np.copy(np.ravel(u))
                    x_new_traj[:, i+j] = np.copy(np.ravel(x))
                    new_J += self.system.calculate_cost(x, u)
                    x = self.system.transition(x, u)
                #x=x+np.random.uniform(low=-5e-3,high=5e-3,size=(30,1))
             
            x_new_traj[:, self.horizon] = np.copy(np.ravel(x))
            new_J += self.system.calculate_final_cost(x)
            if new_J > 1e5:
                iter+=1
                if iter == self.max_iter:
                    print('Max number of iteration for forward pass')
                    return x_up,u_up, J_prev
                new_J=0
                self.alpha=0.8*self.alpha
                x_new_traj = np.zeros((self.system.state_size, self.horizon + 1))
                u_new_traj = np.zeros((self.system.control_size, self.horizon))
                x = np.copy(self.initial_state)
            else:
                adelta_V=self.alpha*self.delta_V1+self.alpha**2*self.delta_V2
                J=0
                for i in range(self.horizon):
                    J+= self.system.calculate_cost(x_up[:,i], u_up[:,i])
                J+= self.system.calculate_final_cost(x_up[:,self.horizon])
            
                z=(J-new_J)/(-np.sum(adelta_V))
                print("z value for the line search: ",z)
                print("Cost of the previous trajectory ",J)
                print("Cost of the new trajectory ",new_J)
                iter+=1
                print("N° of iteration: ",iter)
                if iter == self.max_iter:
                    print('Max number of iteration for forward pass')
                    return x_up,u_up, J_prev
                if (z < 15 and z > 1e-8):
                    x_up=np.copy(x_new_traj)
                    u_up=np.copy(u_new_traj)
                    print("Cost for the new optimal trajectory",new_J)
                    done=1
                    print("Forward pass required: ", iter, " iterations")
                    iter=0
                    return x_up,u_up,new_J
                elif abs(new_J-J) < 0.005:
                    print("Cost for the new optimal trajectory",J)
                    done=1
                    print("Forward pass required: ", iter, " iterations")
                    iter=0
                    return x_up,u_up,J
                else:
                    new_J=0
                    self.alpha=0.8*self.alpha
                    x_new_traj = np.zeros((self.system.state_size, self.horizon + 1))
                    u_new_traj = np.zeros((self.system.control_size, self.horizon))
                    x = np.copy(self.initial_state)
        
        
    def backward_pass(self, lam, mu, x_new,u_new):
        #print("Backward pass")
        self.mu=np.copy(mu)
        self.multipliers=np.copy(lam)
        
        # definition of pn and Pn
        ln_xx = np.copy(self.system.Q_f) 
        ln_x = self.system.Q_f @ (x_new[:, self.horizon] - self.system.goal)
        self.Iu=self.mu[:,self.horizon-1]*np.diag(np.ones(len(self.constraints)))
        C=np.ones(len(self.constraints))
        C_x=np.zeros((len(self.constraints),self.system.state_size))
        for i in range(len(self.constraints)):
            C[i]=self.constraints[i].evaluate_constraint(x_new[:, self.horizon])
            
        # checking if the constraint is active
            if C[i]  < - self.e_constraint and self.multipliers[i,self.horizon-1] == 0: 
                self.Iu[i,i] = 0 # it means that the constraint is not active
            
            C_x[i,:]=self.constraints[i].evaluate_constraint_J(x_new[:, self.horizon])
        cu=np.zeros((len(self.constraints),self.system.control_size)) # no constraints on control trajectory
        pn=ln_x+C_x.T @ (self.multipliers[:,self.horizon-1]+self.Iu @ C)
        Pn=ln_xx+C_x.T @ self.Iu @ C_x
        
        for i in range(self.horizon - 1, -1, -self.fs):
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
                C_x[j,:]=self.constraints[j].evaluate_constraint_J(x)
                C[j]=self.constraints[j].evaluate_constraint(x)
                if C[j]  < - self.e_constraint and self.multipliers[j,i] == 0:
                    self.Iu[j,j] = 0
            if i+1==self.horizon:
                p=pn
                P=Pn
            else:
                p=Q_x + self.K[:,:,i+1].T @ Q_uu @ self.d[:,i+1] + self.K[:,:,i+1].T @ Q_u + Q_ux.T @ self.d[:,i+1]
                P=Q_xx + self.K[:,:,i+1].T @ Q_uu @ self.K[:,:,i+1]+ self.K[:,:,i+1].T @ Q_ux + Q_ux.T @ self.K[:,:,i+1]
   
            A, B = self.system.transition_J(x,u)  #matrices A and B from dynamics
            Q_x = l_xt + A.T @ p + C_x.T @ (self.multipliers[:,i]+self.Iu @ C)
            Q_u = l_ut + B.T @ p + cu.T @ (self.multipliers[:,i]+self.Iu @ C)
            Q_ux = l_uxt + B.T @ P @ A + cu.T @ self.Iu @ C_x
            Q_uu = l_uut + B.T @ P @ B + cu.T @ self.Iu @ cu + self.reg_factor_u * np.identity(self.system.control_size)
            Q_xx = l_xxt + A.T @ P @ A + C_x.T @ self.Iu @ C_x
            
            if np.all(np.linalg.eigvals(Q_uu) > 0):
                for j in range(0,self.fs):
                    self.K[:,:,i-j]=-inv(Q_uu) @ Q_ux
                    self.d[:,i-j]=-inv(Q_uu) @ Q_u
                    self.delta_V1[i-j]=self.d[:,i-j].T @ Q_u
                    self.delta_V2[i-j]=0.5*self.d[:,i-j].T @ Q_uu @ self.d[:,i-j]
            else:
                self.reg_factor_u = self.reg_factor_u*5 
                
    