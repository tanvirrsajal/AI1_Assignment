# main.py
import numpy as np
import matplotlib.pyplot as plt
from JumpyGridWorld import JumpyGridWorld
from QLearningAgent import QLearningAgent
from Plotter import visualizeQValuesOnAxes, visualizePolicyGradeOnAxes, updatePlots, visualizePath
from Globals import rewards, stepsPerEpisode
from Plotter import actionToArrow

def main():
    global rewards, stepsPerEpisode

    np.random.seed(42)

    size = 10
    numObstacles = 5
    stateSize = (size, size)
    actionSize = 5
    learningRate = 0.1
    discountFactor = 0.9
    explorationRate = 0.1
    episodes = 1000

    env = JumpyGridWorld(size, numObstacles)
    agent = QLearningAgent(stateSize, actionSize, learningRate, discountFactor, explorationRate)
    agent.numEpisodes = episodes

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    rewards.clear()
    stepsPerEpisode.clear()
    gridLayouts = []
    performance = []

    print("Training in progress...")

    # Define episodePath before the loop
    episodePath = []

    for episode in range(episodes):
        state = env.reset()
        totalReward = 0
        episodeQValues = []

        # Reset the episode path at the beginning of each episode
        agent.resetEpisodePath()

        while not env.isTerminal(state):
            action = agent.selectAction(state)
            nextState, reward = env.performAction(state, action)
            agent.updateQTable(state, action, reward, nextState)

            state = nextState
            totalReward += reward
            episodeQValues.append(np.copy(agent.qTable[state]))

            # Add the current position to the agent's path
            agent.episodePath.append(state)

        rewards.append(totalReward)
        currentEpisodeLength = len(episodeQValues)
        stepsPerEpisode.append(currentEpisodeLength)

        gridLayout = np.zeros_like(env.state)
        obstaclePositions = list(env.obstacles)
        gridLayout[tuple(zip(*obstaclePositions))] = -1
        gridLayout[env.goalPosition] = 0.5
        gridLayout[state] = 0.8
        gridLayouts.append(gridLayout)

        performance.append(totalReward / currentEpisodeLength)

        # Update episodePath
        episodePath = agent.episodePath

        updatePlots(episode, agent, ax1, ax2, ax3, ax4, ax5, ax6, env, episodePath)

        # Visualize the agent's trajectory path
        visualizePath(size, episodePath, ax6)

    print("Training done.")

    optimalPolicy = np.zeros_like(env.state, dtype=int)
    for i in range(optimalPolicy.shape[0]):
        for j in range(optimalPolicy.shape[1]):
            if gridLayouts[-1][i, j] != -1:
                optimalPolicy[i, j] = np.argmax(agent.qTable[i, j])

    print("\nOptimal Solution:")
    for i in range(optimalPolicy.shape[0]):
        for j in range(optimalPolicy.shape[1]):
            action = optimalPolicy[i, j]
            print(f"State ({i}, {j}): Move {actionToArrow(action)}")

    print("\nOptimal Policy:")
    print(optimalPolicy)

    plt.show()

if __name__ == "__main__":
    main()
