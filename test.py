

from rope_system import Rope
import numpy as np 
from AL_iLQR import AL_iLQR
from constraints import SphereConstraint, CircleConstraint
import matplotlib.pyplot as plt 
if __name__ == '__main__':
    system=Rope(5)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    x0=system.initial_position(0,0,4)
    system.plot_rope(ax,x0)
     
    #print(np.ones((1,601)).shape)
    system.set_cost(0.5*np.identity(system.state_size), 0.05*np.identity(system.control_size))
    Q_f = 10*np.identity(system.state_size)
    for i in range(3*system.n_masses):
        Q_f[i, i] = 150*i
    system.set_final_cost(Q_f)
    
    system.set_goal(np.ravel(system.goal_position(0,0,0)))
    print(system.goal_position(0,0,0))
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    system.plot_rope(ax,system.goal_position(0,0,0))
    #print(system.initial_position(0,0,0))
    horizon=3000
    constraint1=SphereConstraint(system.return_position_actuated_mass(x0), 4, system)
    constraint2=CircleConstraint(np.array([4, 0]), 0.5, system, 12)
    constraint3=CircleConstraint(np.array([3, 0]), 0.5, system, 9)
    solver=AL_iLQR(system, x0, horizon)
    solver.add_constraint(constraint1)
    solver.add_constraint(constraint2)
    solver.add_constraint(constraint3)
    solver.algorithm()
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.plot_rope(ax,solver.x_traj[:,horizon])
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.plot_rope(ax,system.goal)
    print(solver.x_traj[12:15])
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    solver.system.animate_cloth(solver.x_traj, 0.001, gifname="prova")
    
    # print(solver.x_traj)
    # #solver.system.draw_u_trajectories(solver.u_traj)
    # x = plt.subplot(111)
    # plt.scatter(np.arange(100), solver.diff)
    # plt.show()
    