# Aiming
## Auto-Aim

While the results showed some improvements, I felt that my intentionally limited training time was being wasted by the stochastic behavior in the first half of the training. I decided to save some time in training by creating a rudimentary function to provide the bot with perfect aim during the training portions. 

This provides clear examples of successful reward sequences to the agent. Notably, this aimbot only controls behavior when there is an enemy in view. When there are no enemies in view, the action taken is chosen by standard Q-learning. This allows the agent to learn how to behave when there are no enemies on screen, which results in significant increases in performance.

# Defend the Line- Aimbot

## Aimbot On

Using the trained agent, with Aimbot enabled for when target is visible.

[![Watch the video](https://j.gifs.com/XLvxXA.gif)](https://youtu.be/YvPv2kiyKRY)

## Aimbot Off

Using the model trained with Aimbot, but with Aimbot not enabled. All actions taken are learned by the neural network using only imputs from the screen buffer.

[![Watch the video](https://j.gifs.com/3QgRrx.gif)](https://youtu.be/vt5fpE0bzSY)

The performance of the trained agent when not using the aimbot far surpasses that of the agent using the aimbot, as it is able to learn how to aim properly and improve on the aimbot's ability, while the aimbot is stuck in its initial programming. A performance gain can be had by removing the aimbot from the learning process after n number of epochs, allowing more efficient learning. 

![](https://raw.githubusercontent.com/firstlawrobotics/doom-agent/master/autoaim/Save/20200612-212020/20200612-212020_Test_line.png)

 

# Explanation

This is notable for a variety of reasons. First, this is the first time we are seeing above human level performance for the network with no imputs outside of the screen buffer. 

Second, and perhaps more importantly, we show that an aimbot allows extremely high performance for a naive model. Future improvements of the model show clear superhuman performance, as expected. Any real world implementation of a FPS playing agent will require some form of object detection/classifier to identify potential targets. The nets that provide these classifiers will provide bounding boxes that can and should be used for aiming. 

It is not the best use of neural nets to work on solved problems such as aiming and ballistic trajectory, instead they should be focused on learning target prioritization, leading targets, and post-impact corrections. 

Despite this significant increase in performance, the agent has failed to learn ammunition conservation. Further work needs to be done with this before any further progress should be attempted. 

* [Ammo Conservation](../ammo_conservation/ammo-writeup-1.md)
  * Reward Shaping to Influence Agent to Conserve Ammo





