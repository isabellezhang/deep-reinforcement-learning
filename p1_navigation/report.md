# Project 1: Navigation - Report of Learning Algorithm

### Introduction

For this project, we trained an agent with deep Q-Learning algorithm to navigate (and collect bananas!) in a large, square world. Watch below the trained agent:

https://user-images.githubusercontent.com/28734731/156222656-d0202517-8495-46f1-a4bf-ab5ec0de555f.mov

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of our agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Plot of rewards
The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes. The environment was **solved by the deep Q-learning algorithm in 654 episodes** and our trained agent achieved an **average score of 15.03** (over 100 episode). See below a plot of reward per episode: 

![navigation_score_plot](https://user-images.githubusercontent.com/28734731/156225449-a7e93cf9-8bee-42a0-b017-7ac1b0c4af76.png)

### Learning algorithm
In `Navigation_solution.ipynb`, we have a function called `dqn()` which contains a deep Q-learning algorithm used to train our agent. The logic of the algorithm is as follows:
* Initialize `qnetwork_local` (a QNetwork object) to approximate the action value functions with random weights w
* Initialize `qnetwork_target` (a QNetwork object)to approximiate the target action value function with weights w<sup>- </sup>= w
* Initialize replay memory D with capacity `BUFFER_SIZE`.
  * for the episode i_episode from 1 to `n_episodes`:
    * reset the environment and get current state S
    * for time step t from 1 to `max_t`:
      * pick an action A from state S according to Epsilon-greedy policy (with probability epsilon, will randomly select an action and with probability (1-epsilon), will choose the action with highest action value function) 
      * take action A, and observe reward R, next state S' from environment and check if episode has finished `dones`
      * store experience tuple (S, A, R, S') in replay memory D
      * learn every `UPDATE_EVERY` time step
        * Obtain random `minibatch` of tuples (s<sub>j</sub>,a<sub>j</sub>,r<sub>j</sub>,s<sub>j+1</sub>) from D
        * Get `Q_targets_next` -> the max predicted Q value among all possible actions (for next state S') from `qnetwork_target`
        * Compute `Q_targets` for current state, action pair as `rewards + (gamma * Q_targets_next * (1 - dones))`
        * Get `Q_expected` -> expected Q value for current state, action pair from `qnetwork_local`
        * Compute loss as `F.mse_loss(Q_expected, Q_targets)`, minimize loss and use backpropogation to update the weights w in `qnetwork_local`
        * Soft update target network: `θ_target = TAU*θ_local + (1 - TAU)*θ_target`
      *  roll over the state to next time step: S <- S'
  * decrease epsilon by setting `epsilon <- max(eps_end, eps_decay*eps)`

The parameters used in the algorithm are listed as follows:
* n_episodes = 2000 -> maximum number of episodes to train the agent
* max_t = 1000 -> maximum time step in one episode
* BUFFER_SIZE = 100,000 -> experience replay buffer size
* BATCH_SIZE = 64 -> minibatch size
* GAMMA = 0.99 -> discount factor
* TAU = 1e-3 -> for soft update of target parameters
* LR = 5e-4 -> learning rate 
* UPDATE_EVERY = 4 -> how often to update the network
* eps_end = 0.01 -> minimum epsilon in epsilon-greedy policy
* eps_decay = 0.995 -> decay rate for epsilon every episode

There are two other python files that supports our learning algotirhm: 
* dqn_agent.py creates Agent object which interacts and learns from the environment
* model.py creates an QNetwork object which is a neural network used to approximiate the action-value function. It has the following architecture:
  * input layer with 37 nodes (37 states)
  * 1st reLU+linear hidden layer with 74 nodes
  * 2nd reLU+linear hidden layer with 74 nodes
  * output layer with 4 nodes (4 actions)
 
### Ideas of future work/further improvement
1. Double DQN
   - Deep Q-learning tends to overestimate the action values since in the update step of the action values, we always pick the action with maximum action value even in the very early stage where the action values are still very noisy numnbers. To help with this issue, we can use Double Q-learning where we select the best action using our local Q network but use the target Q network to evaluate the action. 
2. Prioritized Experience Replay
   - Deep Q-learning samples uniformly from the replay memory. However, agent may be able to learn more effectively from some transitions than from others. We can use prioritized experience replay to allow these important ones to be sampled with higher probability.
   - We first define TD error as <img width="204" alt="image" src="https://user-images.githubusercontent.com/28734731/156259295-4148eb37-1d58-49b9-b3c3-3f448607b911.png">
   - Then we can define a priority for each experience as <img width="64" alt="image" src="https://user-images.githubusercontent.com/28734731/156259325-06b661e1-dd24-4eff-b17c-b997deea6378.png">
   - The sampling probability for each experience will be <img width="87" alt="image" src="https://user-images.githubusercontent.com/28734731/156259359-a82b119f-d2be-4c83-a1f4-b12ddb049bef.png">
   - The corresponding modified update rule will be <img width="174" alt="image" src="https://user-images.githubusercontent.com/28734731/156259395-c8ed6a7d-dc3c-4c85-81e3-66188de1664e.png">
