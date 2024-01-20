# QLearningAgent.py
import numpy as np

class QLearningAgent:
    def __init__(self, stateSize, actionSize, learningRate, discountFactor, explorationRate):
        # Q-table to store Q-values for each state-action pair
        self.qTable = np.zeros(stateSize + (actionSize,))
        self.learningRate = learningRate  # Learning rate for Q-value updates
        self.discountFactor = discountFactor  # Discount factor for future rewards
        self.explorationRate = explorationRate  # Exploration rate for epsilon-greedy strategy
        self.actionSize = actionSize  # Number of possible actions

    def selectAction(self, state):
        # Implementing action selection logic using epsilon-greedy strategy
        if np.random.rand() < self.explorationRate:
            return np.random.randint(self.actionSize)
        else:
            return np.argmax(self.qTable[state])

    def updateQTable(self, state, action, reward, nextState):
        # Implementing Q-table update logic using the Q-learning update rule
        currentQValue = self.qTable[state + (action,)]
        maxNextQValue = np.max(self.qTable[nextState])

        newQValue = (1 - self.learningRate) * currentQValue + \
                      self.learningRate * (reward + self.discountFactor * maxNextQValue)

        self.qTable[state + (action,)] = newQValue

    def getOptimalPolicy(self, obstacles):
        # Deriving optimal policy based on Q-values
        optimalPolicy = np.zeros(self.qTable.shape[:-1], dtype=int)

        for i in range(optimalPolicy.shape[0]):
            for j in range(optimalPolicy.shape[1]):
                if obstacles[i, j]:
                    optimalPolicy[i, j] = -1  # Mark obstacles
                else:
                    maxQValue = np.max(self.qTable[i, j])
                    candidates = np.where(self.qTable[i, j] == maxQValue)[0]
                    action = np.random.choice(candidates)  # Randomly choose among actions with equal Q-values
                    optimalPolicy[i, j] = action

        return optimalPolicy
