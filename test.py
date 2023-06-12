
from rope_system import Rope
import numpy as np 
from AL_iLQR import AL_iLQR
#import shelve
from constraints import SphereConstraint,BoxConstraint
import matplotlib.pyplot as plt 
if __name__ == '__main__':
    # Initializing the object
    system=Rope(5,0.001) 
    # Initial configuration
    x0=system.initial_position(0,0,4) 
    # Set running cost matrices
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    system.plot_rope(ax,x0)
    plt.savefig("traj.png")
    
    Q=1e-4*np.identity(system.state_size)
    for i in range(3*system.n_masses):
        Q[i, i] = 1e-4 # to change the weight for the position 
    system.set_cost(Q, 0.005*np.identity(system.control_size))
    Q_f = 1e-4*np.identity(system.state_size)
    # Set final cost matrix
    for i in range(3*system.n_masses):
        if i >= (3*system.n_masses -9):
            Q_f[i, i] = 150 # the last three masses have more relevance in minimizing the cost function
    system.set_final_cost(Q_f)
    # Set the final goal
    height=4.2
    system.set_goal(np.ravel(system.goal_position(1,0,height)))
    print(f"The height for the final target is {height}")
    horizon=2000
    name_to_save=1
    # definition of the constraints
    constraint1=SphereConstraint(system.return_position_actuated_mass(x0), 3, system)
    constraint2=BoxConstraint(6.6,3.4,1.1,-1.1,3.3,1.7,system,horizon)
    # starting the solver
    solver=AL_iLQR(system,x0, horizon)
    solver.add_constraint(constraint1)
    solver.add_constraint(constraint2)
    # solver.system.animate_cloth(horizon, solver.x_traj, system.dt, F_history=solver.u_traj, gifname="prova2")

    solver.algorithm()
    print(f"The maximum and minimum values for the control input are: {np.max(solver.u_traj)} and {np.min(solver.u_traj)}")
    print(f"The maximum and minimum values for the velocities are: {np.max(solver.x_traj[15:30,:])} and {np.min(solver.x_traj[15:30,:])}")
    #fig = plt.figure(figsize=(8, 6), dpi=100)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.draw_trajectories(ax,solver.x_traj,solver.x_traj[:,0],solver.x_traj[:,horizon])
    plt.savefig("traj"+str(name_to_save)+"def.png")
    