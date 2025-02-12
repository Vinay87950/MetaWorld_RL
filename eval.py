import torch
import torch.nn as nn
from torch.distributions import Normal
import metaworld
import numpy as np
import imageio
import os
from PIL import Image


'''
Used same as ActorCritic class from main.py
'''

class ActorCritic(nn.Module):
    """
    Actor-Critic Neural Network for Continuous Action Space Reinforcement Learning
    
    This implementation uses separate networks for the actor and critic, with a 
    Gaussian policy for continuous action spaces i.e  probability of taking each possible action.
    The actor outputs the mean of the action distribution, while a learned standard deviation is used to sample actions.
    
    Architecture:Ô
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
        
        # Get mean of action distribution from actor
        mu = self.actor(state)
        
        # Convert log_std to std by exponentiating
        std = self.log_std.exp()
        
        # Create normal distribution with learned parameters
        dist = Normal(mu, std)
        
        return dist, value

def evaluate(env, model_path, num_episodes=10, max_steps=200, save_dir='episode_gifs'):
    '''
    Evaluate the trained model on the environment and save episodes as GIFs
    Args:
        env : Environment object
        model_path : Path to the saved model
        num_episodes : Number of episodes to run
        max_steps : Maximum number of steps per episode
        save_dir : Directory to save the GIF files
    '''
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCritic(state_dim, action_dim)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        frames = []  # Store frames for this episode
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                dist, _ = model(state_tensor)
                action = dist.mean
            
            next_state, reward, terminated, truncated, _ = env.step(action.numpy())
            episode_reward += reward
            state = next_state
            
            # Render and capture frame
            frame = env.render()
            if frame is not None:  # Some environments might return None
                # Convert frame to RGB if necessary
                if len(frame.shape) == 2:  # If grayscale
                    frame = np.stack([frame] * 3, axis=-1)
                frames.append(frame)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode}, Steps: {step+1}, Reward: {episode_reward}")
        
        # Save episode as GIF
        if frames:
            gif_path = os.path.join(save_dir, f'episode_{episode}.gif')
            print(f"Saving GIF to {gif_path}")
            
            # Convert frames to PIL Images if necessary
            pil_frames = []
            for frame in frames:
                # Convert numpy array to PIL Image
                pil_frame = Image.fromarray(frame.astype('uint8'))
                pil_frames.append(pil_frame)
            
            # Save as GIF
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=50,  # Duration between frames in milliseconds
                loop=0
            )

# Initialize MetaWorld environment
mt1 = metaworld.MT1('drawer-close-v2')
env = mt1.train_classes['drawer-close-v2']()

# Set up the specific task instance
task = mt1.train_tasks[0]
env.set_task(task)
env.render_mode = "rgb_array"  # Changed to rgb_array to capture frames

model_path = '/Users/killuaa/Desktop/task2_DRL_Vinay_Kumar/new_models/model_episode_20000.pth'

evaluate(env, model_path)