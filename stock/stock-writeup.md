# Description

In this writeup we do over the fundamnentals of training a FPS playing agent on the basic scenarios in the ViZDoom Reinforcement Learning platform. A standard [Q-Learning algorithm](https://en.wikipedia.org/wiki/Q-learning) is used. 

# Notes

Doom's high framerate allows rapid training, but other environments such as more advanced games or the real world do not provide the ability to iterate at the same speed. It is worthwhile working within these experiental constraints to allow better understanding of what is occuring in the model. All work done in this writeup was conducted with two CPUs in under two hours each run. 


# Basic.wad - Stock Q-Learning

Using the stock Q-learning example (Pytorch implementation) with all standard parameters, the basic scenario where the only actions are move left, move right, and fire, quickly attains near perfect behavior. 

[![](https://j.gifs.com/L7O6nX.gif)](http://www.youtube.com/watch?v=m-DYZ1N2oO8 "Strafing Demo")

* Actions Available:
   * Move Left
   * Move Right
   * Fire
* Rewards:
   * -1 for living
   * -5 for missing
   * +100 for killing target
* Hyper-Parameters:
   * Q-Learning Settings
       * learning_rate = 0.00025
       * discount_factor = 0.99
       * epochs = 30
       * learning_steps_per_epoch = 2000
       * replay_memory_size = 10000
   * NN learning settings
       * batch_size = 64
   * Training regime
       * test_episodes_per_epoch = 100
   * Other parameters
       * frame_repeat = 12

# Basic Turning - Stock Q-Learning

While strafing is intelligent behavior, replacing walking with turning gives us a much more natural looking result. The frame skip was set at 12, which while not a problem while strafing, caused the agent to overshoot and overcorrect, oftentimes repeatedly. This caused suboptimal behavior. This could be rectified by using a smaller frame skip, but was not nessecary for these experiments.

[![](https://j.gifs.com/71m8Yy.gif)](http://www.youtube.com/watch?v=gEkVpXXfXHs "Turning Demo")

* Actions Available:
   * Move Left
   * Move Right
   * Fire
* Rewards:
   * -1 for living
   * -5 for missing
   * +100 for killing target
* Hyper-Parameters: 
   * Varied based on experiment, performance graphs below. Due to the stochastic nature of RL, I should do more experimentation, but this is enough to show some trends. 

### Learning Rate
![Learning Rate](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/stock/Save/bt/basic_turn.cfg%20Learning%20Rate%20Test_line.png)

### Discount Factor
![Discount Factor](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/stock/Save/bt/basic_turn.cfg%20Discount%20Factor%20Test_line.png)

In nearly all experiments, performance quickly converged.

# Defend the Line- Stock Q-Learning

Using the same stock Q-learning, the next scenario has a variety of enemies that move around and attack the agent. Unfortunately, using the short timeline of 25 epochs, we are unable to demonstrate any reasonable results. With the movement of the enemies towards the agent, the probability of a likely hit increases, causing the agent to trust the firing reward far too much. Additionally, due to the lack of penalty for missing, the agent is likely to learn to fire randomly.

[![](https://j.gifs.com/1WO1EV.gif)](https://youtu.be/a2CmaojANks "Defend the Line Stock")

* Actions Available:
   * Move Left
   * Move Right
   * Fire
* Rewards:
   * -1 for living
   * -5 for missing
   * +100 for killing target
* Hyper-Parameters: 
   * Varied based on experiment, performance graphs below. 

### Learning Rate
![Learning Rate](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/stock/Save/dtl_stock/defend_the_line.cfg%20Learning%20Rate%20Test_line.png)

### Discount Factor
![Discount Factor](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/stock/Save/dtl_stock/defend_the_line.cfg%20Discount%20Factor%20Test_line.png)

Performance converges in all cases towards firing blingly straightforward, or rarely, spinning in a circle while firing. Any attempts at modifying hyper-parameters had no effect on improving performance. 

# Defend the Line - Stock Q-Learning, No Moving Enemies

To address the problems caused by moving enemies wandering into the agent's blind firing path, I removed their ability to move. This forced the agent to learn that it needed to turn, but did not teach anything that looked like accuracy. The 12 frame skip continued to limit performance. 

[![](https://j.gifs.com/gZOY8G.gif)](http://www.youtube.com/watch?v=U51vwBFBj2s "Defend the Line Stationary")

Additionally, the unlimited ammo and lack of penalty for missing continued to encourage blind fire. As shown, scores increases a significant amount but did not converge in 25 epochs.

### Learning Rate
![Learning Rate](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/stock/Save/dtl_sta/defend_the_line.cfg%20Learning%20Rate%20Test_line.png)

### Discount Factor
![Discount Factor](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/stock/Save/dtl_sta/defend_the_line.cfg%20Discount%20Factor%20Test_line.png)

# Basic Generalization

Below is a demonstration of basic generalization back to the original problem with moving enemies, using the model trained on stationary enemies. 

[![](https://j.gifs.com/k8kQBN.gif)](http://www.youtube.com/watch?v=dV5MpiGMj8c "Defend the Line Generalized")

Finally, we can also watch the performance of the generalized agent in an environment it has never seen before, Defend the Center.

[![](https://j.gifs.com/p8pwBp.gif)](http://www.youtube.com/watch?v=D5Lgoez-4ok "Defend the Center Generalized")

# Other Writeups:

* [Auto-Aim Training](../autoaim/autoaim-writeup-1.md)
  * Using an Aimbot to Provide Training Data
* [Ammo Conservation](../ammo_conservation/ammo-writeup-1.md)
  * Reward Shaping to Influence Agent to Conserve Ammo
