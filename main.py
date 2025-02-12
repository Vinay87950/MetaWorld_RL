import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from collections import deque
import os
import metaworld


class ActorCritic(nn.Module):
    """
    Actor-Critic Neural Network for Continuous Action Space Reinforcement Learning
    
    This implementation uses separate networks for the actor and critic, with a 
    Gaussian policy for continuous action spaces i.e  probability of taking each possible action.
    The actor outputs the mean of the action distribution, while a learned standard deviation is used to sample actions.
    
    Architecture:
    - Actor: 3-layer neural network outputting action means
    - Critic: 3-layer neural network outputting state-value estimates
    - Both networks use tanh activation functions
    
    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
    
    Network Structure:
        Actor: state_dim -> 64 -> 64 -> action_dim
        Critic: state_dim -> 64 -> 64 -> 1
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Actor network: outputs mean values for the action distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),  # First hidden layer
            nn.Tanh(),                 # Non-linear activation
            nn.Linear(64, 64),         # Second hidden layer
            nn.Tanh(),                 # Non-linear activation
            nn.Linear(64, action_dim)  # Output layer (action means)
        )
        
        # Critic network: estimates state-value function V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),  # First hidden layer
            nn.Tanh(),                 # Non-linear activation
            nn.Linear(64, 64),         # Second hidden layer
            nn.Tanh(),                 # Non-linear activation
            nn.Linear(64, 1)           # Output layer (state value)
        )
        
        # Learnable log standard deviation for action distribution
        # Initialized to zero, will be exponentiated to ensure positive std
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        """
        Forward pass through both actor and critic networks.
        
        Args:
            state (torch.Tensor): Current state input tensor
            
        Returns:
            tuple: (
                Normal distribution object representing the action policy,
                torch.Tensor representing the state value estimate
            )
            
        Notes:
            - The actor outputs parameters for a Gaussian policy
            - std is computed by exponentiating log_std to ensure positivity
            - The returned distribution object can be used to sample actions
              and compute log probabilities for training
        """
        # Get state value estimate from critic
        value = self.critic(state)
        
        # Get mean of action distribution from actor, mean (μ) of a Gaussian distribution
        mu = self.actor(state)
        
        # Convert log_std to std by exponentiating
        std = self.log_std.exp()
        
        # Create normal distribution with learned parameters
        dist = Normal(mu, std)
        
        return dist, value

class PPO:
    """
    Proximal Policy Optimization (PPO) implementation with Generalized Advantage Estimation (GAE)
    
    This implementation includes:
    - GAE for advantage estimation
    - Value function clipping
    - Entropy bonus for exploration
    - Gradient clipping
    
    Args:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        save_dir (str): Directory to save model checkpoints
    """
    def __init__(self, state_dim, action_dim, save_dir='new_models'):
        # Initialize actor-critic network
        self.actor_critic = ActorCritic(state_dim, action_dim)
        
        # Adam optimizer with learning rate 3e-4 (PPO recommended default)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)
        
        # Hyperparameters
        self.gamma = 0.99        # Discount factor for future rewards
        self.gae_lambda = 0.95   # Lambda parameter for GAE
        self.clip_epsilon = 0.2  # PPO clipping parameter, avoids the policy from changing too drastically in one update
        self.c1 = 1.0           # Value function loss coefficient
        self.c2 = 0.01          # Entropy bonus coefficient
        
        # Create directory for saving models
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def compute_gae(self, rewards, values, masks):
        """
        Compute Generalized Advantage Estimation (GAE) using eligibility traces concepts.
        
        GAE combines ideas from TD(λ) and eligibility traces to create a flexible advantage estimator.
        Like eligibility traces, GAE uses an exponentially-weighted average of k-step returns,
        where λ controls the trade-off between bias and variance:
        
        - λ = 0: GAE becomes just the one-step TD error (high bias, low variance)
        - λ = 1: GAE becomes the complete Monte-Carlo return (low bias, high variance)
        
        Mathematical formulation:
        GAE(λ) = Σ(γλ)ᵏ δₜ₊ₖ where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)
        
        Implementation using backward view (similar to eligibility traces):
        1. Start from the last timestep
        2. Accumulate advantages using decay factor (γλ)
        3. Each timestep's advantage influences future estimates with exponentially
           decreasing weight, creating a trace of responsibility
        
        Args:
            rewards (torch.Tensor): Tensor of immediate rewards [r₀, r₁, ..., rₜ]
            values (torch.Tensor): Value estimates for states [V(s₀), V(s₁), ..., V(sₜ)]
            masks (torch.Tensor): Episode termination masks (0 for terminal states)
        
        Returns:
            tuple: (advantages, returns)
                - advantages: GAE-computed advantage estimates
                - returns: Actual discounted returns (advantages + values)
        """
        advantages = torch.zeros_like(rewards) 
        last_advantage = 0 # Keep track of the previous timestep's advantage
        
        # Backward iteration implements eligibility trace-like updates
        for t in reversed(range(len(rewards))):
            # Handle terminal states (next_value = 0 if terminal)
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            
            # Compute TD error (δₜ = rₜ + γV(sₜ₊₁) - V(sₜ))
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            
            # Update advantage estimate using eligibility trace update rule:
            # Aₜ = δₜ + (γλ)Aₜ₊₁
            # This creates an exponentially-weighted sum of future TD errors
            advantages[t] = delta + self.gamma * self.gae_lambda * masks[t] * last_advantage 
            last_advantage = advantages[t]
        
        # Compute actual returns by adding advantages to value estimates
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, masks):
        """
        Update policy and value function using PPO algorithm
        
        Key steps:
        1. Convert data to tensors
        2. Compute advantages using GAE
        3. Normalize advantages
        4. Perform multiple epochs of updates with minibatches
        5. Compute PPO policy loss with clipping
        6. Compute value function loss
        7. Add entropy bonus for exploration
        8. Update network parameters with clipped gradients
        
        Args:
            states (array): States from rollout
            actions (array): Actions taken
            old_log_probs (array): Log probabilities of actions under old policy
            rewards (array): Rewards received
            masks (array): Episode termination masks
        """
        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = torch.FloatTensor(rewards)
        masks = torch.FloatTensor(masks)

        # Compute advantages and returns
        dist, values = self.actor_critic(states)
        advantages, returns = self.compute_gae(rewards, values.detach(), masks)
        
        # Normalize advantages (reduces variance of training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Perform multiple epochs of updates
        for _ in range(10):  # Number of epochs
            # Get current policy distribution and value estimates
            dist, current_values = self.actor_critic(states)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().mean()
            
            # Compute PPO policy loss with clipping
            ratio = torch.exp(log_probs - old_log_probs)  # Importance sampling ratio
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value function loss
            value_loss = 0.5 * (returns - current_values.squeeze()).pow(2).mean()
            
            # Compute total loss (negative because we're minimizing)
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Perform gradient update with clipping
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

    def save_model(self, episode):
        """
        Save model and optimizer state to disk
        
        Args:
            episode (int): Current episode number for filename
        """
        path = os.path.join(self.save_dir, f'model_episode_{episode}.pth')
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

def train(env, agent, max_episodes=50000):
    """
    Main training loop for PPO agent in a reinforcement learning environment.
    
    This function implements the following steps for each episode:
    1. Reset environment and initialize episode variables
    2. Collect trajectory data by running the current policy
    3. Update policy using collected experiences
    4. Save model checkpoints periodically
    
    Args:
        env: Gym-like environment with step() and reset() methods
        agent: PPO agent instance containing actor-critic network
        max_episodes (int): Maximum number of training episodes
    
    Implementation Details:
    - Collects complete episodes before performing PPO updates
    - Stores trajectory data (states, actions, rewards, etc.) in lists
    - Uses the agent's policy to sample actions and compute log probabilities
    - Tracks episode rewards for monitoring training progress
    """
    for episode in range(max_episodes):
        # Reset environment and get initial state
        state = env.reset()[0]  # [0] index gets state from gym's reset() return
        done = False
        episode_reward = 0
        
        # Initialize lists to store trajectory data
        states = []      # Store states for updating
        actions = []     # Store actions taken
        rewards = []     # Store rewards received
        log_probs = []   # Store log probabilities of actions
        masks = []       # Store done flags (0 for terminal states)
        
        # Collect trajectory data
        while not done:
            # Convert state to tensor for neural network
            state_tensor = torch.FloatTensor(state)
            
            # Get action distribution from current policy
            dist, _ = agent.actor_critic(state_tensor)
            
            # Sample action from distribution
            action = dist.sample()
            
            # Compute log probability of the sampled action
            # sum(-1) combines log probs if action is multi-dimensional
            log_prob = dist.log_prob(action).sum(-1)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.detach().numpy())
            
            # Check if episode is done (either terminated or truncated)
            done = terminated or truncated
            
            # Store trajectory data
            states.append(state)
            actions.append(action.detach().numpy())
            rewards.append(reward)
            log_probs.append(log_prob.detach())
            masks.append(1-done)  # 1 for non-terminal states, 0 for terminal
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # End of episode processing
            if done:
                # Update policy using collected trajectory
                agent.update(states, actions, log_probs, rewards, masks)
                
                # Save model checkpoint every 100 episodes
                if (episode + 1) % 100 == 0:
                    agent.save_model(episode + 1)
                
                # Print episode statistics
                print(f"Episode {episode}, Reward: {episode_reward}")
                break

"""
Setup and training script for MetaWorld drawer-close task using PPO

MetaWorld is a benchmark environment for meta-reinforcement learning,
containing various robotic manipulation tasks. This script sets up the
drawer-close task and trains a PPO agent to solve it.
"""

# Initialize MetaWorld environment
mt1 = metaworld.MT1('drawer-close-v2') 
env = mt1.train_classes['drawer-close-v2']()  

# Set up the specific task instance
task = mt1.train_tasks[0]  # Get first task configuration
env.set_task(task)         # Configure environment with task parameters

# Get environment dimensions for network architecture
state_dim = env.observation_space.shape[0]   # Dimensionality of state space (robot + drawer state)
action_dim = env.action_space.shape[0]       # Dimensionality of action space (robot joint actions)

# Initialize PPO agent
agent = PPO(state_dim, action_dim) 

# Start training
train(env, agent)

"""
Reference - took help from chatgpt as there was no github solution for present for these tasks 
1. Chatgpt --- 'https://chatgpt.com/share/67a8974e-a81c-8006-b003-d9db7c11512c'

2. Meta World env - 'https://meta-world.github.io/' 

"""