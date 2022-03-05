# Project 2: Continuous Control - Report of Learning Algorithm

### Introduction of environment 

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

https://user-images.githubusercontent.com/28734731/156841356-abcb5b76-3361-452e-afde-c8407be804c9.mov

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Plot of rewards
In order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes. The environment was **solved by the DDPG algorithm in 32 episodes** and our trained agent achieved an **average score of 37.06** (over 100 episode). See below a plot of reward per episode: 

![reward plot](https://user-images.githubusercontent.com/28734731/156842799-8c166d42-a744-48f4-b1b3-49c52ebe6652.png)


### Learning algorithm
The code present in `Continuous_control_solution.ipynb`, `model.py`, and  `ddpg_agent.py` are modified based on ddpg framework from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum

In `Continuous_Control_solution.ipynb`, we have a function called `ddpg()` which contains a ddpg algorithm used to train our agent. The logic of the algorithm is as follows:
* Initialize a local, target Actor netowrk and a local, target Critic Network 
* Initialize replay memory D with capacity `BUFFER_SIZE`.
* Initialize Ornstein-Uhlenbeck noise process 
  * for the episode i_episode from 1 to `n_episodes`:
    * reset the environment and get current state S
    * for time step t from 1 to `max_t`:
      * pick an action A from state S according to local Actor network for each agent with added noise
      * take action A for each agent, and observe reward R, next state S' from environment and check if episode has finished `dones`
      * store experience tuple (S, A, R, S') in replay memory D for each agent after each time step
      * learn `NUM_UPDATES` times after every `UPDATE_EVERY` time step
        * Obtain predicted next-state action and Q values (`Q_targets_next`) from target Actor and Critic model
        * Compute `Q_targets` for current state, action pair as `rewards + (gamma * Q_targets_next * (1 - dones))`
        * Get `Q_expected` -> expected Q value for current state, action pair from local Critic model
        * Compute loss as `F.mse_loss(Q_expected, Q_targets)`, minimize loss and use backpropogation to update the weights in local Critic network
        * Obtain predicted current state action (`actions_pred`) from local Actor model
        * Compute loss as negative of predicted Q value of current state, actions_pred pair from local Critic model,  minimize loss and use backpropogation to update the weights in local actor network
        * Soft update target actor and critic network: `θ_target = TAU*θ_local + (1 - TAU)*θ_target`
      *  roll over the state to next time step: S <- S'

The parameters used in the algorithm are listed as follows:
* n_episodes = 1000 -> maximum number of episodes to train the agent
* max_t = 2000 -> maximum time step in one episode
* BUFFER_SIZE = int(1e6) -> replay buffer size
* BATCH_SIZE = 128 -> minibatch size
* GAMMA = 0.99 -> discount factor
* TAU = 1e-3 -> for soft update of target parameters
* LR_ACTOR = 3e-4 -> learning rate of the actor 
* LR_CRITIC = 3e-4 -> learning rate of the critic
* WEIGHT_DECAY = 0 -> L2 weight decay
* UPDATE_EVERY = 20 -> update the actor and critic networks every UPDATE_EVERY timestpes
* NUM_UPDATES = 10 -> update the actor and critic networks NUM_UPDATES times after every UPDATE_EVERY timesteps

There are two other python files that supports our learning algotirhm: 
* ddpg_agent.py creates Agent object which interacts and learns from the environment
* model.py creates a actor and critic network. The actor network maps states to actions and Critic network maps state, action pair to Q-values.
* The actor network has the following architecture:
  * input layer with 33 nodes (33 states)
  * 1st reLU+linear hidden layer with 256 nodes
  * 2nd reLU+linear hidden layer with 128 nodes
  * output layer with tanh activation function of 4 nodes (action size)
* The critic network has the following architecture:
  * input layer with 33 nodes (33 states)
  * 1st reLU+linear hidden layer with 256+4 nodes
  * 2nd reLU+linear hidden layer with 128 nodes
  * output layer with 1 node

The major modifications that have been made to the code in order to train in the reacher environment are the following:
1. Adapt the code to work with Unity environemnt
2. Modify the code to work with 20 agents (following suggestions by Udacity in Benchmark Implementation)
   * make sure each agent adds its experience to a replay buffer that is shared by all agents
   * update the actor and critic networks 10 times after every 20 timesteps
3. Use gradient clipping when training the critic network (following suggestion by Udacity in Benchmark Implementation)
4. Reduce number of neurons in the Actor and Critic network to 256, 128 in 1st and 2nd hidden layer
   * This would speed up the learning process while still supporting effective learning
4. Increase noise added to the actions to encourage exploration during training
   * increase sigma to 0.3
   * add in both negative and postive random noise `dx = self.theta * (self.mu - x) + self.sigma * np.array([2*random.random()-1 for i in range(len(x))])`
5. Extend max timestep for each episode to 2000 from 300
   * This allows the agents to have more time to learn during each episode since 300 is too short for effective learning

### Ideas of future work/further improvement
1. Prioritized Experience Replay
2. Try out PPO, A2C, A3C algorithms 
