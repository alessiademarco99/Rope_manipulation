# First_attempt_with_jit

The goal state is the first mass in position (0,0,0) and the last one in position (4,0,0).\n
I started with a simple circle constraint on x and y position of the last two masses to see the behaviour of the algorithm, since I already had them implemented.\n
The jit function is applied to the transition of the state and the speed of the algorithm is improved a lot.\n

Next steps:\n
- Add box constraints and contact model for it\n
- Tunes the parameter for Q, Q_f and R because the behaviour changes a lot\n
- Add arrow for the force applied to the first mass in the final plot\n
