# Project 3: Collaboration and Competition - Report of Learning Algorithm

### Introduction of environment

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.


https://user-images.githubusercontent.com/28734731/157286105-31785e50-8e64-45ec-8426-d8cdbe9392a2.mov


In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

### Plot of rewards
In order to solve the environment, the agent must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). The environment was **solved by the MADDPG algorithm in 1300 episodes** and our trained agent achieved an **average score of +0.63** (over 100 episode). See below a plot of reward per episode: 

![score plot](https://user-images.githubusercontent.com/28734731/157290133-10ec0364-0245-40ec-933c-1cb411174d60.png)


### Learning algorithm
The code present in `Tennis_solution.ipynb`, `model.py`, and  `ddpg_agent.py` are modified based on ddpg framework from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

We first take the solution for P2 - continous control and adpated the code so that it works for multiagent condition. 
In `Tennis_solution.ipynb`, we have a function called `maddpg()` which contains a maddpg algorithm used to train our agent. The logic of the algorithm is as follows:
* Initialize a local, target Actor netowrk and a local, target Critic Network for each agent (2 agents in total)
* Initialize replay memory D<sub>i</sub> with capacity `BUFFER_SIZE` for each agent i.
* Initialize Ornstein-Uhlenbeck noise process for each agent
  * for the episode i_episode from 1 to `n_episodes`:
    * reset the environment and get current states S
    * for time step t from 1 to `max_t`:
      * pick an action A<sub>i</sub>  from state S<sub>i</sub> according to local Actor network for each agent i with added noise
      * take action A<sub>i</sub> for each agent, and observe rewards R, next states S' from environment and check if episode has finished `dones`
      * store experience tuple (S<sub>i</sub>, A<sub>i</sub>, R<sub>i</sub>, S'<sub>i</sub>) in replay memory D<sub>i</sub> for each agent after each time step
      * learn `NUM_UPDATES` times after every `UPDATE_EVERY` time step for each agent i
        * Obtain random `minibatch` of tuples (s<sub>j</sub>,a<sub>j</sub>,r<sub>j</sub>,s<sub>j+1</sub>) from D<sub>i</sub> 
        * Obtain predicted next-state action and Q values (`Q_targets_next`) from target Actor and Critic model
        * Compute `Q_targets` for current state, action pair as `rewards + (gamma * Q_targets_next * (1 - dones))`
        * Get `Q_expected` -> expected Q value for current state, action pair from local Critic model
        * Compute loss as `F.mse_loss(Q_expected, Q_targets)`, minimize loss and use backpropogation to update the weights in local Critic network
        * Obtain predicted current state action (`actions_pred`) from local Actor model
        * Compute loss as negative of predicted Q value of current state, actions_pred pair from local Critic model,  minimize loss and use backpropogation to update the weights in local actor network
        * Soft update target actor and critic network: `θ_target = TAU*θ_local + (1 - TAU)*θ_target`
      *  roll over the state to next time step: S <- S'

The parameters used in the algorithm are listed as follows:
* n_episodes = 3000 -> maximum number of episodes to train the agent
* max_t = 10000 -> maximum time step in one episode
* BUFFER_SIZE = int(1e6) -> replay buffer size
* BATCH_SIZE = 128 -> minibatch size
* GAMMA = 0.99 -> discount factor
* TAU = 6e-3 -> for soft update of target parameters
* LR_ACTOR = 3e-4 -> learning rate of the actor 
* LR_CRITIC = 3e-4 -> learning rate of the critic
* WEIGHT_DECAY = 0 -> L2 weight decay
* UPDATE_EVERY = 1 -> update the actor and critic networks every UPDATE_EVERY timestpes
* NUM_UPDATES = 1 -> update the actor and critic networks NUM_UPDATES times after every UPDATE_EVERY timesteps

There are two other python files that supports our learning algotirhm: 
* ddpg_agent.py creates Agent object which interacts and learns from the environment
* model.py creates a actor and critic network. The actor network maps states to actions and Critic network maps state, action pair to Q-values.
* The actor network has the following architecture:
  * input layer with 24 nodes (24 states)
  * 1st reLU+linear hidden layer with 256 nodes
  * 2nd reLU+linear hidden layer with 128 nodes
  * output layer with tanh activation function of 2 nodes (action size)
* The critic network has the following architecture:
  * input layer with 24 nodes (24 states)
  * 1st reLU+linear hidden layer with 256+2 nodes
  * 2nd reLU+linear hidden layer with 128 nodes
  * output layer with 1 node

### Ideas of future work/further improvement
1. Prioritized Experience Replay
2. Try out PPO, A2C, A3C algorithms 
