# First_attempt_with_jit

The goal state is the first mass in position (0,0,0) and the last one in position (4,0,0).
I started with a simple circle constraint on x and y position of the last two masses to see the behaviour of the algorithm, since I already had them implemented.
The jit function is applied to the transition of the state and the speed of the algorithm is improved a lot.

Next steps:
- Add box constraints and contact model for it - done
- Tunes the parameter for Q, Q_f and R because the behaviour changes a lot - done
- Add arrow for the force applied to the first mass in the final plot - done

- compare the behaviour by changing the timestep (0.001 - 0.003 - 0.005 - 0.01) 
