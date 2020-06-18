## Assumed limitations:

Limited number of available training episodes. Doom's high framerate allows incredible amounts of training, but other environments such as more advanced First Person Shooters or the real world do not provide the ability to iterate at the same speed. 


# Basic.wad - Stock Q-Learning

Using the stock Q-learning example (Pytorch implementation) with all standard parameters, the basic scenario where the only actions are move left, move right, and fire, quickly attains near perfect behavior. 


## Actions Available:
* Move Left
* Move Right
* Fire

## Rewards:
* -1 for living
* -5 for missing
* +100 for killing target

## Hyper-Parameters:
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

[![](https://j.gifs.com/L7O6nX.gif)](http://www.youtube.com/watch?v=m-DYZ1N2oO8 "Strafing Demo")

<test line graph > 


# Basic_turn.wad - Stock Q-Learning

While strafing is intelligent behavior, we are better off turning and firing at enemies. Replacing walking with turning gives us a much more natural looking result.

## Actions Available:
* Turn Left
* Turn Right
* Fire

## Rewards:
* -1 for living
* -5 for missing
* +100 for killing target


## Hyper-Parameters: 
Varied based on experiment, performance graphs below. (Due to stochastic nature of RL, I should do more experimentation, but this is enough to show some trends.

The frame skip was set at 12, which while not a problem while strafing, caused the agent to overshoot and overcorrect, oftentimes repeatedly. This caused suboptimal behavior. This could be rectified by using a smaller frame skip, but was not nessecary.

[![](http://img.youtube.com/vi/Gnqm0H2Nvec/0.jpg)](http://www.youtube.com/watch?v=Gnqm0H2Nvec "Turning Demo 1")

[![](http://img.youtube.com/vi/gEkVpXXfXHs/0.jpg)](http://www.youtube.com/watch?v=gEkVpXXfXHs "Turning Demo 2")


<discount_factor>

<learning_rate>


# Defend_the_line.wad - Stock Q-Learning

Using the same stock Q-learning, the next scenario has a variety of enemies that move around and attack the agent. Unfortunately, using the short timeline of 25 epochs, we are unable to demonstrate any reasonable results. With the movement of the enemies towards the agent, the probability of a likely hit increases, causing the agent to trust the firing reward far too much. 

## Actions Available:
* Turn Left
* Turn Right
* Fire

## Rewards:
* +1 for killing target

## Hyper-Parameters: 
Varied based on experiment, performance graphs below. 


[![](http://img.youtube.com/vi/gEkVpXXfXHs/0.jpg)](http://www.youtube.com/watch?v=gEkVpXXfXHs "Defend the Line Stock")

<discount_factor>
<learning_rate>


# Defend_the_line.wad - Stock Q-Learning, No Moving Enemies

To address the problems caused by moving enemies wandering into the agent's blind firing path, I removed their ability to move. This forced the agent to learn that it needed to turn, but did not teach anything that looked like accuracy. The 12 frame skip continued to limit performance. 

[![](http://img.youtube.com/vi/U51vwBFBj2s/0.jpg)](http://www.youtube.com/watch?v=U51vwBFBj2s "Defend the Line Stationary")


Additionally, the unlimited ammo and lack of penalty for missing continued to encourage blind fire. 

<discount_factor>
<learning_rate>

Below is a demonstration of basic generalization back to the original problem with moving enemies, using the model trained on stationary enemies. 

[![](http://img.youtube.com/vi/dV5MpiGMj8c/0.jpg)](http://www.youtube.com/watch?v=dV5MpiGMj8c "Defend the Line Generalized")

Finally, we can also watch the performance of the generalized agent in an environment it has never seen before, Defend the Center.

[![](http://img.youtube.com/vi/D5Lgoez-4ok/0.jpg)](http://www.youtube.com/watch?v=D5Lgoez-4ok "Defend the Center Generalized")
