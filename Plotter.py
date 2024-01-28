# Plotter.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from Globals import rewards, stepsPerEpisode

def actionToArrow(action):
    # Map action index to arrow symbol
    mapping = {
        0: '↑',  # Up
        1: '↓',  # Down
        2: '←',  # Left
        3: '→',  # Right
        4: '⇆'   # Jump
    }
    return mapping.get(action, '?')  # Return '?' if action is not in the mapping

def visualizePath(size, path, ax):
    # Visualize the agent's trajectory path
    ax.clear()
    ax.set_title(r'$\bf{Agent\'s\ Trajectory\ Path}$', fontsize=14)
    ax.set_xlabel('Grid Width')
    ax.set_ylabel('Grid Height')
    ax.set_xticks(np.arange(size+1))  # Add +1 to include the edge for better visualization
    ax.set_yticks(np.arange(size+1))  # Add +1 to include the edge for better visualization
    ax.grid(which='both', color='black', linestyle='-', linewidth=2)

    path_x, path_y = zip(*path)
    ax.plot(np.array(path_y)+0.5, np.array(path_x)+0.5, marker='o', linestyle='-', color='blue')  # Add +0.5 to center the markers in the cells
    ax.invert_yaxis()  # Invert y-axis to match the visualization of the grade


def visualizeQValuesOnAxes(qValues, ax):
    # Visualizing Q-values as a heatmap
    ax.clear()
    bestQValues = np.max(qValues, axis=2)
    cax = ax.matshow(bestQValues, cmap='cool')
    cbar = plt.colorbar(cax, ax=ax)
    cbar.ax.set_ylabel('Q-values', rotation=-90, va="bottom")
    ax.set_title('Q-values Heatmap', pad=20)
    ax.set_xlabel('Grid Width')
    ax.set_ylabel('Grid Height')
    ax.set_xticks(np.arange(qValues.shape[1]))
    ax.set_yticks(np.arange(qValues.shape[0]))

def visualizePolicyGradeOnAxes(qValues, ax):
    # Visualizing policy grades
    ax.clear()

    policy = np.argmax(qValues, axis=2)

    actionValues = {
        0: 0.25,  # Up
        1: 0.50,  # Down
        2: 0.75,  # Left
        3: 1.0,   # Right
        4: 0.0    # Jump
    }

    policyValues = np.vectorize(actionValues.get, otypes=[float])(policy)

    cax = ax.matshow(policyValues, cmap='cool', aspect='equal')

    cmap = plt.cm.cool
    norm = Normalize(vmin=0.0, vmax=1.0)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.set_yticklabels(['Jump', 'Up', 'Down', 'Left', 'Right'])  # Text labels

    for i, row in enumerate(policy):
        for j, action in enumerate(row):
            arrow = actionToArrow(action)
            ax.text(j, i, arrow, ha='center', va='center', fontsize=12, color='black')

    ax.set_xticks(np.arange(policy.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(policy.shape[0]) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    ax.set_title('Policy Grade Visualization\n')

def updatePlots(episode, agent, ax1, ax2, ax3, ax4, ax5, ax6, env, episodePath):
    global rewards, stepsPerEpisode  # Global variables
    if episode == agent.numEpisodes - 1:  # For showing plots only at the end of training
        visualizeQValuesOnAxes(agent.qTable, ax5)
        visualizePolicyGradeOnAxes(agent.qTable, ax4)
        ax4.set_title(r'$\bf{Policy\ Grade\ Visualization}$', fontsize=14)
        ax5.set_title(r'$\bf{Q-values\ Heatmap}$', fontsize=14)
        ax6.set_title(r'$\bf{Agent\'s\ Trajectory\ Path}$', fontsize=14)


        # Plotting steps per episode
        ax1.plot(range(episode + 1), stepsPerEpisode, color='blue')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title(r'$\bf{Steps\ per\ Episode}$', fontsize=14)

        # Plotting rewards per episode
        ax2.plot(range(episode + 1), rewards, color='magenta')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title(r'$\bf{Rewards\ per\ Episode}$', fontsize=14)

        # Plotting cumulative reward per episode
        cumulative_rewards = np.cumsum(rewards)
        ax3.plot(range(episode + 1), cumulative_rewards, color='green')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Cumulative Reward')
        ax3.set_title(r'$\bf{Cumulative\ Reward\ per\ Episode}$', fontsize=14)

        # Visualize the agent's trajectory path
        visualizePath(env.size, episodePath, ax6)

        # Adding some space between the rows and columns of plots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
