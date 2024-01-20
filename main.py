# main.py
import numpy as np
import matplotlib.pyplot as plt
from JumpyGridWorld import JumpyGridWorld
from QLearningAgent import QLearningAgent
from Plotter import visualizeQValuesOnAxes, visualizePolicyGradeOnAxes, updatePlots
from Globals import rewards, stepsPerEpisode
from Plotter import actionToArrow

def main():
    global rewards, stepsPerEpisode  # Global variables

    np.random.seed(42)  # Set a random seed for reproducibility

    # Parameters
    size = 10
    numObstacles = 5
    stateSize = (size, size)
    actionSize = 5  # Including jump action
    learningRate = 0.1
    discountFactor = 0.9
    explorationRate = 0.1
    episodes = 1000

    # Initializing environment and Q-learning agent
    env = JumpyGridWorld(size, numObstacles)
    agent = QLearningAgent(stateSize, actionSize, learningRate, discountFactor, explorationRate)
    agent.numEpisodes = episodes  # Storing the number of episodes for reference

    # Creating a single figure for visualization with 4 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Lists to store data for plotting
    rewards.clear()  # Clearing the local list, use the global one
    stepsPerEpisode.clear()  # Clearing the local list, use the global one
    gridLayouts = []
    performance = []

    # Training loop
    print("Training in progress...")
    for episode in range(episodes):
        state = env.reset()
        totalReward = 0
        episodeQValues = []

        while not env.isTerminal(state):
            action = agent.selectAction(state)
            nextState, reward = env.performAction(state, action)
            agent.updateQTable(state, action, reward, nextState)

            state = nextState
            totalReward += reward
            episodeQValues.append(np.copy(agent.qTable[state]))

        rewards.append(totalReward)
        currentEpisodeLength = len(episodeQValues)
        stepsPerEpisode.append(currentEpisodeLength)

        # Grid Layout
        gridLayout = np.zeros_like(env.state)
        obstaclePositions = list(env.obstacles)
        gridLayout[tuple(zip(*obstaclePositions))] = -1
        gridLayout[env.goalPosition] = 0.5
        gridLayout[state] = 0.8
        gridLayouts.append(gridLayout)

        # Performance
        performance.append(totalReward / currentEpisodeLength)

        # Update plots for each episode
        updatePlots(episode, agent, ax1, ax2, ax3, ax4, ax5, ax6)
        

    print("Training done.")

    # Deriving optimal policy
    optimalPolicy = np.zeros_like(env.state, dtype=int)
    for i in range(optimalPolicy.shape[0]):
        for j in range(optimalPolicy.shape[1]):
            if gridLayouts[-1][i, j] != -1:
                optimalPolicy[i, j] = np.argmax(agent.qTable[i, j])

    # Printing results
    print("\nOptimal Solution:")
    for i in range(optimalPolicy.shape[0]):
        for j in range(optimalPolicy.shape[1]):
            action = optimalPolicy[i, j]
            print(f"State ({i}, {j}): Move {actionToArrow(action)}")
    
    #Showing optimal policy in array
    print("\nOptimal Policy:")
    print(optimalPolicy)

    # Showing the final plots at the end
    plt.show()

if __name__ == "__main__":
    main()

