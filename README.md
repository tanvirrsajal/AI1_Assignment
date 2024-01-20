# Introduction:

This project implements a Q-learning algorithm for a Jumpy Grid World environment, where an agent navigates a grid layout with obstacles and a goal position. The agent can perform actions, including a unique jump action, and learns to optimize its path to maximize cumulative rewards. The code comprises modular components such as JumpyGridWorld, QLearningAgent, and Plotter. The training loop in Main.py orchestrates the learning process, updating Q-values and collecting data for visualizations. Plots include steps per episode, rewards per episode, cumulative rewards, exploration decay, Q-values heatmap, and policy grade visualization. The implementation allows for easy parameter adjustments and presents opportunities for improvement, such as fine-tuning parameters, exploring different grid sizes, enhancing exploration strategies, refining visualizations, and considering parallelization for larger environments. Overall, the project provides a comprehensive exploration of Q-learning in a dynamic grid world setting with detailed visualizations for analysis and interpretation.

## Installation:

To run the project, one needs to have Python3 installed. Additionally, install the required libraries using the following command:

```bash
pip install numpy matplotlib
```

Make sure that all project files (main.py, JumpyGridWorld.py, etc.) are residing in the same directory.
Running the Project:

To experience the magic of reinforcement learning in action, execute the Main.py script:

```bash
python3 main.py
```

Trouble Shooting:

In case of facing problem, consider the following:

    ModuleNotFoundError: If this crops up, ensure the required dependencies are installed using the provided pip install command.

    Compatibility Issues: Make sure you are using a Python version that harmonizes with the code.

    File Not Found: Confirm that all essential files are present in the same directory.

## Architechture and components:
### JumpyGridWorld.py:
This file serves as the bedrock of our environment. The JumpyGridWorld class encapsulates the world, defining methods for resetting, checking positions, obstacle avoidance, and action execution.

### QLearningAgent.py:
The QLearningAgent class is the brain behind our agent. It handles action selection, Q-table updates, and the derivation of the optimal policy.

### Plotter.py:
Plotter.py injects life into our data, providing functions to visualize Q-values and policy grades. It's the visual storyteller of our agent's learning journey.

### Globals.py:
This file hosts global variables (rewards and stepsPerEpisode) as scorekeepers during our agent's training.

### Main.py:
The main protagonist orchestrates the entire narrative. It initializes the environment, sets up the Q-learning agent, runs the training loop, and finally, unveils the visual story through captivating plots.

## Plot Details:
1. Steps per Episode Plot (ax1):

Objective: Showcase the evolution of the number of steps taken by the agent in each episode.

Interpretation: A dwindling trend implies the agent's growing adeptness at navigating the environment efficiently.
2. Rewards per Episode Plot (ax2):

Objective: Illuminate the total reward garnered by the agent in each episode.

Interpretation: A rising trajectory signifies the agent's triumphant grasp of the optimal policy.
3. Cumulative Reward per Episode Plot (ax3):

Objective: Visualize the cumulative sum of rewards obtained by the agent.

Interpretation: An upward slope indicates the agent consistently achieving higher cumulative rewards.
4. Exploration Decay Plot (ax4):

Objective: Depict the decline in exploration rate over episodes.

Interpretation: A gradual descent hints at the agent's transition from exploration to exploitation as it hones in on the optimal policy.
5. Q-values Heatmap Plot (ax5):

Objective: Illuminate the Q-values across the grid world, offering insights into the learned values for each state-action pair.

Interpretation: Brighter regions signify higher Q-values, reflecting the agent's learned preferences for certain actions in specific states.
6. Policy Grade Visualization Plot (ax6):

Objective: Visualize the agent's policy by displaying the optimal action (arrow) for each grid cell.

Interpretation: Arrows indicate the agent's preferred actions at each location. A cohesive pattern of arrows unveils a learned policy for navigating the grid world.

## How it Works:

    - The agent explores the environment using an epsilon-greedy strategy, updating Q-values based on rewards and future Q-values.

    - Plots are updated at the end of each episode to visualize the learning progress.

    - The training loop continues for a specified number of episodes.

    - The optimal policy is derived from the learned Q-values.

    - Results, optimal solution, and optimal policy are printed.

Potential Improvements:

    - Dynamic Environment: Allow dynamic changes in the environment during training.

    - Deep Q-Learning: Implement a deep Q-learning approach for handling complex policies.

    - More Complex Environments: Introduce more diverse grid worlds with varying sizes, shapes, and obstacles.

    - Parallelization: Consider parallelizing the training loop for faster learning.
