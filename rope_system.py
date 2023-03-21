import jax 
import jax.numpy as np
from jax import jit, vmap, grad, value_and_grad, jacfwd
from jax.config import config
config.update("jax_debug_nans", True)
jax.config.update('jax_enable_x64', True)
import time
from functools import partial
from skspatial.objects import Sphere
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from IPython.display import HTML
matplotlib.rc('animation', html='jshtml')
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Arrow3D(FancyArrowPatch):
    # Arrow plotting code from: 
    # https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


class DynamicalSystem:
    def __init__(self, state_size, control_size):
        self.state_size = state_size
        self.control_size = control_size
        
    def set_cost(self, Q, R):
		# one step cost = x.T * Q * x + u.T * R * u
        self.Q = Q
        self.R = R
        
    def get_cost(self):
        return self.Q, self.R
        
    def set_final_cost(self, Q_f):
        self.Q_f=Q_f
    
    def calculate_cost(self, x, u):
        x=np.ravel(x)
        return 0.5*((x-self.goal).T.dot(self.Q).dot(x-self.goal)+u.T.dot(self.R).dot(u))
    
    def calculate_final_cost(self,x):
        x=np.ravel(x)
        return 0.5*(x-self.goal).T.dot(self.Q_f).dot(x-self.goal)
    
    def set_goal(self, x_goal):
        self.goal = x_goal
        

class Rope(DynamicalSystem):
    def __init__(self,n_masses,dt):
        super().__init__(n_masses*6,3)
        self.dt = dt
        self.control_bound = np.ones(self.control_size) * 100
        self.goal = np.zeros(self.state_size)
        self.n_masses=n_masses
        self.l_rest=1
        self.m=0.3/self.n_masses
        self.k_elastic=10.0
        self.k_shear = 5.0          
        self.k_bend = 1          
        self.c_elastic = self.k_elastic / self.n_masses  # N / (m/s)
        self.c_shear = self.k_shear / self.n_masses      # N / (m/s)
        self.c_bend = self.k_bend / self.n_masses        # N / (m/s)
        g = 9.80665         # m/s^2
        self.ag = np.array([0.0, 0.0, -g])  # acceleration due to gravity
        self.actuated_masses = [0]
    
    def init_position(self,xa,ya,za,i):
        xb = xa
        yb = ya
        zb = za-self.l_rest*i
        return np.array([xb, yb, zb], dtype=np.float32)
    
    def initial_position(self,x,y,z):
        p0 = np.array([self.init_position(x,y,z,i) for i in np.arange(self.n_masses)])
        p0=p0.reshape(self.n_masses*3,1)
        v0 = np.zeros((self.n_masses*3, 1))
        s0 = np.vstack((p0, v0))
        return s0
    
    def goal_pos(self,xa,ya,za,i):
        xb = xa+self.l_rest*i
        yb = ya
        zb = za
        return np.array([xb, yb, zb], dtype=np.float32)
    
    def goal_position(self,x,y,z):
        p0 = np.array([self.goal_pos(x,y,z,i) for i in np.arange(self.n_masses)])
        p0=p0.reshape(self.n_masses*3,1)
        v0 = np.zeros((self.n_masses*3, 1))
        s0 = np.vstack((p0, v0))
        return s0
        
    def return_position_actuated_mass(self,s):
        p,_=np.vsplit(s,2)
        p=p.reshape(self.n_masses,3)
        return p[0,:]
    
    def return_position_last2mass(self,s):
        p,_=np.vsplit(s,2)
        p=p.reshape(self.n_masses,3)
        return p[self.n_masses-2:self.n_masses,:]
    
    #@jit
    def hooke_damped(self,p,v):
        f = np.zeros_like(p)
        for i in range(0, self.n_masses-1):
            # Compute the displacement and distance between neighboring particles
            delta_pos = p[i+1,:] - p[i,:]
            delta_vel = v[i+1,:] - v[i,:]
            #print("delta pos",delta_pos)
            
            dist = np.linalg.norm(delta_pos)
            
            dist_non_zero=np.maximum(dist, self.l_rest/10)
            
            u= delta_pos / dist_non_zero
            x=(dist - self.l_rest) * u
            x_dot=np.dot(delta_vel,u)*u
            # Compute the force exerted by the spring
            # f_spring = k_elastic * u + k_shear * u + k_bend *u + c_elastic*x_dot + c_shear *x_dot + c_bend * x_dot
            f_spring = self.k_elastic * x + self.c_elastic*x_dot + self.k_shear * x +  self.c_shear*x_dot 
            f=f.at[i].add(f_spring)
            f=f.at[i+1].add(-f_spring)
            
            if i < self.n_masses-2:
                delta_pos = p[i+2,:] - p[i,:]
                delta_vel = v[i+2,:] - v[i,:]
                dist = np.linalg.norm(delta_pos)
            
                dist_non_zero=np.maximum(dist, 2*self.l_rest/10)
                
                u= delta_pos / dist_non_zero
                x=(dist - 2*self.l_rest) * u
                x_dot=np.dot(delta_vel,u)*u
                f_bend = self.k_bend * x + self.c_bend*x_dot  
                f=f.at[i].add(f_bend)
                f=f.at[i+2].add(-f_bend)
            # Add the force to the particles
            
            
        return f
    
    #@jit
    def change_of_state(self,s, F_act):
        s=s.reshape(6*self.n_masses,1)
        p,v=np.vsplit(s,2)
        p=p.reshape(self.n_masses,3)
        
        v=v.reshape(self.n_masses,3)
        
        # z = p[:, 2]
        # jumpv=np.where(z <= 0, 0, v[:,2])
        # jumpp=np.where(z <= 0, 0, z) #non lo fa
        #print(jump.shape)
        
        F = self.hooke_damped(p, v)
        
        #F += ground_force(p)
        # v=v.at[:,2].set(jumpv)
        # p=p.at[:,2].set(jumpp)
        
        F = F.at[np.index_exp[self.actuated_masses, :]].add(F_act)
        
        a = F / self.m
        a += self.ag
        a=a.reshape(self.n_masses*3,1)
        v=v.reshape(self.n_masses*3,1)
        return np.vstack((v, a))
    
    # #@partial(jit)
    # def transition(self,x,u):
    #     x=x.reshape(6*self.n_masses,1)
    #     f = self.change_of_state
    #     k1 = f(x, u)
    #     k2 = f(x + k1 * (self.dt / 2),u)
    #     k3 = f(x + k2 * (self.dt / 2), u)
    #     k4 = f(x + k3 * self.dt, u)
    #     x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) * (self.dt / 6)
    #     return x_new
    
    @partial(jit, static_argnums=(0,1,))
    def RK4(self, f, x, u):
        k1 = f(x, u)
        k2 = f(x + k1 * (self.dt / 2),u)
        k3 = f(x + k2 * (self.dt / 2),u)
        k4 = f(x + k3 * self.dt,u)
        x_new = x + (k1 + 2 * k2 + 2 * k3 + k4) * (self.dt / 6)
        return x_new


    def transition(self,x,u):
        x=x.reshape(6*self.n_masses,1)
        f=self.change_of_state
        x_new=self.RK4(f,x,u)
        p,v=np.vsplit(x_new,2)
        p=p.reshape(self.n_masses,3)
        v=v.reshape(self.n_masses,3)
        #floor
        z = p[:, 2]
        jumpv=np.where(z <= 0, 0, v[:,2])
        jumpp=np.where(z <= 0, 0, z)
        v=v.at[:,2].set(jumpv)
        p=p.at[:,2].set(jumpp)
        
        #box
        for k in range(self.n_masses):
            if p[k,0] >= 3.4 and p[k,0] <=6.6:
                if p[k,1] >= -0.6 and p[k,1] <=0.6:
                    if p[k,2] >= 1.9 and p[k,2] <=3.28:
                        if np.isclose(p[k,2],1.9,atol=1e-1):
                            v=v.at[k,2].set(-0.8*v[k,2])
                            p=p.at[k,2].set(1.9)
                        elif np.isclose(p[k,2],3.2,atol=1e-1):
                            v=v.at[k,2].set(0.0)
                            p=p.at[k,2].set(3.3)
                        if np.isclose(p[k,0],3.5,atol=1e-1):
                            v=v.at[k,0].set(-0.8*v[k,0])
                            p=p.at[k,0].set(3.4)
                # elif np.isclose(p[k,2],4.2,atol=1e-1):
                #     v=v.at[k,2].set(0.0)
                #     p=p.at[k,2].set(4.3)
        p=p.reshape(self.n_masses*3,1)
        v=v.reshape(self.n_masses*3,1)
        x_new=np.vstack((p,v))
        return x_new
    
    #@jit
    def transition_J(self,x,u):
        # print("x_transition: ", x.shape)
        # print(x)
        x_temp=x.reshape(6*self.n_masses,1)
        #u=u.reshape(3,1)
        dynamics_jac_state = jacfwd(self.transition, argnums=0)(x_temp,u)
        # print("after first")
        dynamics_jac_control = jacfwd(self.transition, argnums=1)(x_temp,u)
        A = dynamics_jac_state[:,1,:,1]
        B = dynamics_jac_control[:,1,:]
        # print(x)
        return A,B
    
    def plot_rope(self, ax, s, 
               xlim=[-2, 4], 
               ylim=[-2, 3], 
               zlim=[0, 4]):
        ux=6.5
        uy=0.5
        uz=3.2
        lx=3.5
        ly=-0.5
        lz=2.0
        
        p,_=np.vsplit(s,2)
        p=p.reshape(self.n_masses,3)
        x, y, z = np.transpose(p)
        ax.clear()  # necessary for the animations
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        # circle1 = plt.Circle((4, 0), 0.5, color=(0, 0.8, 0.8))
        # circle2 = plt.Circle((3, 0), 0.5, color=(0, 0.8, 0.8))
        #ax.add_artist(circle1)
        # ax.add_patch(circle1)
        # ax.add_patch(circle2)
        # #ax.add_artist(circle2)
        # art3d.pathpatch_2d_to_3d(circle1, z=4, zdir="z")
        # art3d.pathpatch_2d_to_3d(circle2, z=4, zdir="z")
        ax.scatter(x, y, z, c='red', s=10)
        ax.plot(x,y,z,c='blue')
        # x = [3.5,6.5,6.5,3.5],[3.5,6.5,6.5,3.5],[3.5,3.5,3.5,3.5],[3.5,3.5,6.5,6.5],[3.5,3.5,6.5,6.5],[6.5,6.5,6.5,6.5]
        # y = [-0.5,-0.5,0.5,0.5],[-0.5,-0.5,0.5,0.5],[-0.5,-0.5,0.5,0.5],[-0.5,-0.5,-0.5,-0.5],[0.5,0.5,0.5,0.5],[-0.5,-0.5,0.5,0.5]
        # z = [3.8,3.8,3.8,3.8],[4.2,4.2,4.2,4.2],[3.8,4.2,4.2,3.8],[3.8,4.2,4.2,3.8],[3.8,4.2,4.2,3.8],[3.8,4.2,4.2,3.8]
        x = [lx,ux,ux,lx],[lx,ux,ux,lx],[lx,lx,lx,lx],[lx,lx,ux,ux],[lx,lx,ux,ux],[ux,ux,ux,ux]
        y = [ly,ly,uy,uy],[ly,ly,uy,uy],[ly,ly,uy,uy],[uy,uy,uy,uy],[uy,uy,uy,uy],[ly,ly,uy,uy]
        z = [lz,lz,lz,lz],[uz,uz,uz,uz],[lz,uz,uz,lz],[lz,uz,uz,lz],[lz,uz,uz,lz],[lz,uz,uz,lz]

        surfaces = []

        for i in range(len(x)):
            surfaces.append( [list(zip(x[i],y[i],z[i]))] )

        for surface in surfaces:
            ax.add_collection3d(Poly3DCollection(surface, facecolors='cyan', linewidths=1, edgecolors='b', alpha=.2))

        sphere = Sphere([0,0,4],3)
        sphere.plot_3d(ax, alpha=0.2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
      
    def plot_arrow(self,ax, start, force, scale):
        end = start + scale * force
        x0, y0, z0 = start
        x1, y1, z1 = end
        a = Arrow3D([x0, x1], [y0, y1], [z0, z1], mutation_scale=10, lw=3, arrowstyle="-|>", color="mediumseagreen", zorder=10)
        ax.add_artist(a)  
        
    def animate_cloth(self, horizon, s_history, dt, fps=60, F_history=None, force_scale=0.2, gifname=None):
        fig = plt.figure(figsize=(5, 5), dpi=100)
        fig.subplots_adjust(0,0,1,1,0,0)
        ax = fig.add_subplot(111, projection='3d')
        plt.close()  # prevents duplicate output 

        fps_simulation = 1 / dt
        skip = np.floor(fps_simulation / fps).astype(np.int32)
        fps_adjusted = fps_simulation / skip
        print('fps was adjusted to:', fps_adjusted)
        horizon=horizon-1

        def animate(i):
            j = min(i * skip, horizon)
            #print(j)
            p = s_history[:,j].reshape(self.n_masses*6,1)
            #print(p)
            self.plot_rope(ax, p)
            #ax.text2D(0.1, 0.9, 't = {:.3f}s'.format(j * dt), transform=plt.transAxes)
            if F_history is not None:
            #     for mass_id, F in zip(self.actuated_masses, F_history):
                self.plot_arrow(ax, np.ravel(p[0:3]), np.ravel(F_history[:,j]), force_scale) 


        n_frames = (horizon) // skip + 1  # this +1 is for the initial frame
        if not (horizon) % skip == 0:
            n_frames += 1  # this +1 is to ensure the final frame is shown

        anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000*dt*skip)
        #plt.show()
        #writergif = animation.PillowWriter(fps=fps_adjusted) 
        if gifname is not None:
            anim.save(gifname + '.gif', writer='imagemagick')

        return anim
            
    def draw_trajectories(self, x_trajectories,p0):
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        self.plot_rope(p0)
        plt.plot(x_trajectories[0, 0::5], x_trajectories[1, 0::5], 4,color='r')
        ax.set_aspect("equal")
        # ax.set_xlim(0, 3)
        # ax.set_ylim(0, 3)
        plt.grid()
        plt.show()
        
        
if __name__ == '__main__':
    system=Rope(5)
    x0=system.initial_position(-1,2,0.5)
    # print(x0.shape)
    print(system.initial_position(0,0,0))
    
    system.plot_rope(system.initial_position(0,0,0))
    # F_act0=np.array([0.0,0.0,5.0])
    # x_new=system.transition(x0,F_act0)
    # A,B=system.transition_J(x_new,F_act0)
    # # print(A.shape)
    # system.plot_rope(x_new)
    # plt.show()
    # #F_act0=np.array([0.0,-0.0,0.0])
    # for i in range(1000):
    #     x_new=system.transition(x_new,F_act0)
    #     if i % 25 == 0:
    #         system.plot_rope(x_new)
        #print(x_new)
        