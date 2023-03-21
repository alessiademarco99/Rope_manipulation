# First_attempt_with_jit

The goal state is the first mass in position (2,0,4.2) and the last one in position (6,0,4.2).
The jit function is applied to the transition of the state and the speed of the algorithm is improved a lot.
The upper face of teh box constraint is in z=3.2. I'm trying with different heights for the goal position to see how the algorithm behaves (3.2 - 3.8 - 4 - 4.2) and also different starting positions.

Next steps:
- Add box constraints and contact model for it - done
- Tunes the parameter for Q, Q_f and R because the behaviour changes a lot - done
- Add arrow for the force applied to the first mass in the final plot - done

- compare the behaviour by changing the timestep (0.001 - 0.003 - 0.005 - 0.01) 
