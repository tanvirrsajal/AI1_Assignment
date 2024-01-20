# Project summary:
This project implements a Q-learning algorithm for a Jumpy Grid World environment, where an agent navigates a grid layout with obstacles and a goal position. The agent can perform actions, including a unique jump action, and learns to optimize its path to maximize cumulative rewards. The code comprises modular components such as JumpyGridWorld, QLearningAgent, and Plotter. The training loop in main.py orchestrates the learning process, updating Q-values and collecting data for visualizations. Plots include steps per episode, rewards per episode, cumulative rewards, exploration decay, Q-values heatmap, and policy grade visualization. The implementation allows for easy parameter adjustments and presents opportunities for improvement, such as fine-tuning parameters, exploring different grid sizes, enhancing exploration strategies, refining visualizations, and considering parallelization for larger environments. Overall, the project provides a comprehensive exploration of Q-learning in a dynamic grid world setting with detailed visualizations for analysis and interpretation.

## Installation:
To run the project, one needs to have Python3 installed. Additionally, install the required libraries using the following command:
```bash
pip install numpy matplotlib
```
Make sure that all project files (main.py, JumpyGridWorld.py, etc.) are residing in the same directory.
Running the Project:
To run the program, First clone the project and then execute the main.py script(one at a time).

```bash
git clone https://github.com/tanvirrsajal/AI1_Assignment.git
cd AI1_Assignment
python3 main.py
```

## Trouble Shooting:
In case of facing problem, consider the following:
- ModuleNotFoundError: If this crops up, ensure the required dependencies are installed using the provided pip install command.
- Compatibility Issues: Make sure you are using a Python version that harmonizes with the code.
- File Not Found: Confirm that all essential files are present in the same directory.

## Architechture and components:
### JumpyGridWorld.py:
The JumpyGridWorld.py module defines the environment where the Q-learning agent learns. The JumpyGridWorld class initializes a grid world with obstacles and provides methods for resetting the environment, checking valid positions, detecting obstacles, and performing actions. The jump action introduces a random element, enhancing the complexity of the learning task.

### QLearningAgent.py:
The QLearningAgent.py script encapsulates the Q-learning agent's logic. The QLearningAgent class maintains a Q-table to store Q-values for each state-action pair. It features methods for selecting actions using an epsilon-greedy strategy, updating the Q-table based on rewards and future Q-values, and deriving the optimal policy. The agent's learning rate, discount factor, and exploration rate are adjustable parameters.

### Plotter.py:
Plotter.py focuses on visualization functions to aid in understanding the Q-learning agent's behavior. The module utilizes Matplotlib to create heatmaps for Q-values and policy grades, with functions like visualizeQValuesOnAxes and visualizePolicyGradeOnAxes. The updatePlots function dynamically updates plots during training, showcasing the progression of rewards, steps per episode, Q-values, and policy grades.

### Globals.py:
Globals.py serves as a simple utility file, declaring global variables rewards and stepsPerEpisode. These variables store cumulative rewards and steps per episode throughout the training process. By making them global, different parts of the project can easily access and update this shared data.

### Main.py:
Main.py orchestrates the entire reinforcement learning process. It first initializes the JumpyGridWorld environment with specified parameters such as grid size, number of obstacles, and action size. Then, a Q-learning agent is instantiated. The training loop iterates through episodes, updating the Q-table based on the agent's actions and the environment's responses. Data such as rewards, steps per episode, grid layouts, and performance metrics are collected for visualization. The script concludes by deriving and printing the optimal policy based on the trained Q-values and showcasing various plots representing the training progress.

## Flowchart
https://github.com/tanvirrsajal/AI1_Assignment/blob/master/JumpyGridWorld.pdf

## Plot Details:
### Steps per Episode:
*Objective:* Showcase the evolution of the number of steps taken by the agent in each episode.
*Interpretation:* A dwindling trend implies the agent's growing adeptness at navigating the environment efficiently.

### Rewards per Episode:
*Objective:* Illuminate the total reward garnered by the agent in each episode.
*Interpretation:* A rising trajectory signifies the agent's triumphant grasp of the optimal policy.

### Cumulative Reward per Episode Plot:
*Objective:* Visualize the cumulative sum of rewards obtained by the agent.
*Interpretation:* An upward slope indicates the agent consistently achieving higher cumulative rewards.

### Exploration Decay:
*Objective:* Depict the decline in exploration rate over episodes.
*Interpretation:* A gradual descent hints at the agent's transition from exploration to exploitation as it hones in on the optimal policy.

### Q-values Heatmap:
*Objective:* Illuminate the Q-values across the grid world, offering insights into the learned values for each state-action pair.
*Interpretation:* Brighter regions signify higher Q-values, reflecting the agent's learned preferences for certain actions in specific states.

### Policy Grade Visualization:
*Objective:* Visualize the agent's policy by displaying the optimal action (arrow) for each grid cell.
*Interpretation:* Arrows indicate the agent's preferred actions at each location. A cohesive pattern of arrows unveils a learned policy for navigating the grid world.

## How it Works:
- The agent explores the environment using an epsilon-greedy strategy, updating Q-values based on rewards and future Q-values.
- Plots are updated at the end of each episode to visualize the learning progress.
- The training loop continues for a specified number of episodes.
- The optimal policy is derived from the learned Q-values.
- Results, optimal solution, and optimal policy are printed.

## Result
The agent successfully learns to navigate the grid world and reach the goal state. The visualizations demonstrate the agent's improved performance over time. The Q-values heatmap shows that the agent learns to prioritize paths with higher expected rewards. The policy grade visualization shows that the agent becomes more confident in its chosen actions. The steps per episode plot shows a decrease in steps as the agent learns, indicating more efficient navigation. The rewards per episode plot shows an increase in reward as the agent learns, indicating better reward maximization. The cumulative rewards plot shows a consistent increase in reward throughout the training process.

## Conclusion
This project successfully demonstrates the application of reinforcement learning to solve a navigation task. The Q-learning algorithm enabled the agent to learn from its experiences and make optimal decisions in a dynamic environment. 

## Customization
Parameters inside the main.py can be modified to see the change in the agent's learning process and performance. 

## Potential Improvements:
- *Dynamic Environment:* Allow dynamic changes in the environment during training.
- *Deep Q-Learning:* Implement a deep Q-learning approach for handling complex policies.
- *More Complex Environments:* Introduce more diverse grid worlds with varying sizes, shapes, and obstacles.
- *Parallelization:* Consider parallelizing the training loop for faster learning.
