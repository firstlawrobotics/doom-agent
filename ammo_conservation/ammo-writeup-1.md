
# Ammunition Conservation
## Defend_the_center.wad Reward Tests

Defend_the_line has a limited amount of ammo, and in theory, the limiting factor for the agent should be the number of rounds that it has. However, because there is no negative reward for missing, the only way for the agent to learn that would be by realizing over time that its score is dependent on ammo conservation, and with how stochastic the initial training is, that will never occur on this limited time scale with no chance of convergence. 

One approach to handling this would be to create a negative reward for firing and missing using the .wad action scripts. 

To determine an efficient reward I ran some quick tests modifying the ratio of points awarded for a kill versus a miss. 10/-1, 5/-1, 2/-1, 3/-2, 1/-1.

Reusing the most effective hyper-parameters from the stock tests, we ran experiments for each of the ratios. Results are shown below: 


