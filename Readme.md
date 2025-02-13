# MetaWorld Drawer-Close Task using PPO

## Project Overview
This project implements Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) to solve the drawer-close task in MetaWorld. The implementation includes a complete actor-critic architecture, PPO training loop, and GAE computation.

## About the environment
Environment: MetaWorld
MetaWorld is a benchmark environment for meta-reinforcement learning research, specifically focused on robot manipulation tasks. This project uses the MT1 (single-task) variant with the drawer-close-v2 task.

## Drawer-Close Task
- Objective: Train a robotic arm to close an open drawer
- State Space: Robot joint positions and drawer state
- Action Space: Continuous control of robot joint movements
- Reward: Based on the robot's success in closing the drawer

## Algorithm: Proximal Policy Optimization (PPO)
PPO is a policy gradient method that uses clipping to prevent too large policy updates, making training more stable.

- **Key Components :**
  - Actor-Critic Architecture
  - GAE (Generalized Advantage Estimation) that combines TD(λ) with eligibility traces


- **How does it works ? :**
  - Collecting trajectories using current policy
  - Computing advantages using GAE
  - Updating policy and value function using PPO
  - Repeating until desired performance is achieved


- **What does the agent learns to do? :**
  - Move the robot arm efficiently
  - Approach the drawer
  - Execute closing motion
 
- **How to use this code ? :**

1. So use this code - first create a conda env by - "conda create mujoco"

2. activate the env - "conda activate mujoco"

3. Then to install the env run "pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld"

4. The folder will be downloaded as Metaworld 

5. cd Metaworld 

6. pip install metaworld 

7. pip install 'mujoco-py<2.2,>=2.1'

8. Then test the env if working on not named as "test_env.py"

9. Run "main.py" for the complete training 

10. For evaluation use "eval.py"



## Results 
**30000 episodes:**<br><br>
  <img src="https://github.com/Vinay87950/MetaWorld_RL/blob/main/episode_gifs/episode_1.gif">


