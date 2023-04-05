

from rope_system import Rope
import numpy as np 
from AL_iLQR import AL_iLQR
from constraints import SphereConstraint,BoxConstraint
import matplotlib.pyplot as plt 
if __name__ == '__main__':
    system=Rope(5,0.001) 
    fig = plt.figure(figsize=(8, 6), dpi=100) 
    ax = fig.add_subplot(111, projection='3d')
    x0=system.initial_position(0,0,4)
    system.plot_rope(ax,x0)
    #print(np.ones((1,601)).shape)
    system.set_cost(1e-4*np.identity(system.state_size), 0.005*np.identity(system.control_size))
    Q_f = 1e-4*np.identity(system.state_size)
    for i in range(3*system.n_masses):
        if i >= (3*system.n_masses - 9):
            Q_f[i, i] = 150 #more important the last ones 
    system.set_final_cost(Q_f)
    
    system.set_goal(np.ravel(system.goal_position(1,0,4.2)))
    print(system.goal_position(1,0,4.2))
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    system.plot_rope(ax,system.goal_position(2,0,3.2))
    #print(system.initial_position(0,0,0))
    horizon=2000
    
    constraint1=SphereConstraint(system.return_position_actuated_mass(x0), 5, system)
    constraint2=BoxConstraint(6.5,3.5,1,-1,3.2,2.0,system,horizon)
    
    solver=AL_iLQR(system, x0, horizon)
    solver.add_constraint(constraint1)
    solver.add_constraint(constraint2)
    solver.algorithm()
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.plot_rope(ax,solver.x_traj[:,horizon])
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.plot_rope(ax,system.goal)
    print(solver.x_traj[:,horizon])
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.animate_cloth(horizon, solver.x_traj, system.dt, F_history=solver.u_traj, gifname="prova2")
    print(np.max(solver.u_traj))
    print(np.min(solver.u_traj))
    
    # print(solver.x_traj)
    # #solver.system.draw_u_trajectories(solver.u_traj)
    # x = plt.subplot(111)
    # plt.scatter(np.arange(100), solver.diff)
    # plt.show()
    