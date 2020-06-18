## Assumed limitations:

Limited number of available training episodes. Doom's high framerate allows incredible amounts of training, but other environments such as more advanced First Person Shooters or the real world do not provide the ability to iterate at the same speed. 


## Basic.wad - Stock Q-Learning

Using the stock Q-learning example (Pytorch implementation) with all standard parameters, the basic scenario where the only actions are move left, move right, and fire, quickly attains near perfect behavior. 

Video of Initial Random Behavior at 0 Epochs:
< video of random at 0>

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

<video of 25 episodes>

<test line graph > 


## Basic_turn.wad - Stock Q-Learning

While strafing is intelligent behavior, we are better off turning and firing at enemies. Replacing walking with turning gives us a much more natural looking result.

Video of Initial Random Behavior at 0 Epochs:
< video of random at 0>


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

<video of 30 episodes bt2>


<discount_factor>

<learning_rate>

<batch_size>


## Defend_the_line.wad - Stock Q-Learning

Using the same stock Q-learning, the next scenario has a variety of enemies that move around and attack the agent. Unfortunately, using the short timeline of 25 epochs, we are unable to demonstrate any reasonable results. With the movement of the enemies towards the agent, the probability of a likely hit increases, causing the agent to trust the firing reward far too much. 

Video of Initial Random Behavior at 0 Epochs:
< video of random at 0>

## Actions Available:
* Turn Left
* Turn Right
* Fire

## Rewards:
* +1 for killing target

## Hyper-Parameters: 
Varied based on experiment, performance graphs below. 


<video of 30 episodes straight firing>
<video of 30 episodes spinning>

<discount_factor>
<learning_rate>
<batch_size>

## Defend_the_line.wad - Stock Q-Learning, No Moving Enemies

To address the problems caused by moving enemies wandering into the agent's blind firing path, I removed their ability to move. This forced the agent to learn that it needed to turn, but did not teach anything that looked like accuracy. The 12 frame skip continued to limit performance. 

<video of 30 episodes rotating>

Additionally, the unlimited ammo and lack of penalty for missing continued to encourage blind fire. 

# Aiming
## Auto-Aim

While the results showed some improvements, I felt that my intentionally limited training time was being wasted by the stochastic behavior in the first half of the training. I decided to save some time in training by creating a rudimentary function to provide the bot with perfect aim during the training portions. 

This provides clear examples of successful reward sequences to the agent. Notably, this aimbot only controls behavior when there is an enemy in view. When there are no enemies in view, the action taken is chosen by standard Q-learning. This allows the agent to learn how to behave when there are no enemies on screen, which results in significant increases in performance.

### Defend_the_line

< video of random at 0>


Training Scores:

With Aimbot
< video of train at 25>

Testing Scores: 

Using the model trained with Aimbot, but Aimbot not enabled. 
< video of test at 25>

This is notable for a variety of reasons. First, this is the first time we are seeing above human level performance for the network with no imputs outside of the screen buffer. 

Second, and perhaps more importantly, we show that an aimbot allows extremely high performance for a naive model. Future improvements of the model show clear superhuman performance, as expected. Any real world implementation of a FPS playing agent will require some form of object detection/classifier to identify potential targets. The nets that provide these classifiers will provide bounding boxes that can should be used for aiming. It is not the best use of neural nets to work on solved problems such as aiming and ballistic trajectory, instead they should be focused on learning target prioritization, leading targets, and post-impact corrections. 

Despite this significant increase in performance, the agent has failed to learn ammunition conservation. Further work needs to be done with this before any further progress should be attempted. 





