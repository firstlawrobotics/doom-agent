# First Law Robotics

This repo contains the code and writeups in support of First Law Robotics. 

# Writeups

[Stock ViZDoom Scenarios](./stock/stock-writeup.md)

# Easier Things to Do

### Cover and Concealment

How do you train an RL agent to understand their and the enemy's use of cover and concealment? By using the Ray-Casting Minimap reward system I have developed, I expect to be able to train a decent understanding of exposure that will significantly modify the behavior of agents towards something more human-like. 

Requirements: 

* Fully trained model to bootstrap a fresh model 
* Ray-Cast Minimap VizDoom Implementation
* Intend to start with Defend the Circle to prove it works before moving on to Deathmatch WADs

### [Bootstrapped Training](./autoaim/autoaim-writeup-1.md)

Using a rudimentary auto-aim agent to bootstrap experience rapidly allows the training of an agent that is significantly better than the auto-aim agent. This is very useful for saving time in training, and the methods to use saved experience should be formalized. 

I want to use pre-trained Arnold examples to bootstrap deathmatch experience, but use a completely fresh network and reward system (namely the Ray-Cast Minimap). 


### Infrared Targeting

Using the different layers available you can make a pseudo IR camera to train IR targeting and test to see how effective it is. Because Computer Vision will fail it will be a completely different problem. Having the ability to switch to an IR trained network is likely useful for a variety of situations.



# Hard Problems 


### [Ammo Conservation](./ammo_conservation/ammo-writeup-1.md)

Attempts so far to teach one shot one kill have failed miserably. Presumably I should train a functional agent and then modify rewards to heavily penalize missing. 

Requirements: 

* Fully trained model to bootstrap a fresh model 
* Iterate through reward functions until one repeatedly hits


# Rules of Engagement and Civilian Discrimination

How do you train an RL agent to follow a dynamically set Rules of Engagement? Probably by manually setting the Rules of Engagement. Avoiding non-combatants is an incredibly complex task, but by building out different rules sets for different ROEs and cycling through them, without relying on RL for discrimination, it might be somewhat possible. 

Requirements: 

* Fully trained model to bootstrap a fresh model 
* Custom WAD for an ROE Deathmatch Mode
   * Only get points for killing allowed enemy
   * Lose points for killing anything not allowed by current ROE
   * ROE Dynamically changes throughout

To get networks that can handle these different ROEs, bootstrap the following ROEs.

###  Target Anything that Moves 

For pure sci-fi badness, how do you train an RL agent to shoot anything that moves?

Requirements: 

* Fully trained model to bootstrap a fresh model 
* Custom WADs where only specific enemies move
* Reward Function that recognizes difference between enemy types


###  Target Only Those Firing Weapons

How would we only fire on enemies with weapons that are actively engaged in firing? An easier way to do this is with my custom YOLO dataset that applies the firing tag to enemies who display the firing sprite.

Requirements: 

* Fully trained model to bootstrap a fresh model 
* Custom WADs where only specific enemies fire
* Custom YOLO Dataset with 'Firing' label
* Reward Function that recognizes difference between enemy types

###  Target those With Weapons

How to only fire on enemies with Weapons? This relies on custom YOLO. An easier way to do this is with my custom YOLO dataset that applies the firing tag to enemies who have a gun. This can be done with the custom YOLO dataset OR with a gun detection YOLO dataset.

Requirements: 

* Fully trained model to bootstrap a fresh model 
* Custom WADs where only specific enemies fire
* Custom YOLO Dataset with 'Weapon' label OR Public YOLO Weapon dataset
* Reward Function that recognizes difference between enemy types

# Human in the Loop/On the Loop

Is it possible for an agent with a human in the loop to beat an agent running headless? Is it possible for 'On the Loop' to have better discrimination in an ROE Deathmatch?

# Surrender

How do you train an RL agent to demand and accept a surrender? How do you request to surrender? Does that even make sense?

# Camouflage and Partially See-Through Cover

In an agent heavily reliant on computer vision, camouflage and partial obscurants will significantly impact confidence levels and thus, effectiveness. How do you train against camo and cover that only displays small amounts of the enemy? I believe IR/ visual mixing will be very useful/required for any sort of effective agent. There are no partially transparent cover items in Doom, but I think windows/doors/strange shaped walls with enemies on the other side will do a decent job of simulating that effect. For camo, matching enemy textures to walls should do a good job of forcing the agent to adapt. 





