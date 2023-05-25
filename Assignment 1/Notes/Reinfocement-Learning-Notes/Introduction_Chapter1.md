
- [Introduction](#introduction)
- [Elements of reinforcement learning](#elements-of-reinforcement-learning)
  - [Policy](#policy)
  - [Reward Signal](#reward-signal)
  - [Value function](#value-function)
  - [Model](#model)


# Introduction
Notes taken from Reinforcement Leaerning an Introduction by Richard S. Sutton and Andrew G. Barto

Two essential components: 
1. Trial and error
2. Delayed Reward

A Rainforcement learning agent should be able to interact over time with the enviroment.\
The problem of reinforcement learning is formalized by dynamical systemm: optimal control of the incompletely known Markov decision process. \
Goal state of the enviroment is also needed

# Elements of reinforcement learning 
The four sub-elements od reinforcement learning are:\
1.  Policy
2.  Rewad Signal
3.  Value function
4.  Model (Optional)

## Policy 
It defines the learning agents way of behaving at a given time.  Can be understood as the mapping of state of the enviroment to the action to be taken.
## Reward Signal
Defines the goal of the learnig problem. \
The enviroment sends the reward signal, and the agent triies to maximize said reward signal.\
The reward signal is the key in changing or alterig the agents policy. \ One could think of this as the cost of travelling from one node to another (think of A* search).


## Value function
This unlike the reward is not the immediate response. Rather, it is the estimated reward in the long run.
\
One could think of the value function as a heuristic, i.e. think of th heuristic used in intelligent searches such as A* search or best first search.

## Model 
This an optional part of reinforcement learning.
\
RL problems that use models are known as Model-based methods. Model-free methods are trial and error methods. 
\
There are even some hybrid approaches.\
The model mimicks the enviroment. They are used to make inferences on the enviroment will behave on the certain policy. 








