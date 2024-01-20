# JumpyGridWorld.py
import numpy as np

class JumpyGridWorld:
    def __init__(self, size, numObstacles):
        self.size = size
        self.state = np.zeros((size, size))
        self.startPosition = (0, 0)
        self.goalPosition = (size - 1, size - 1)
        self.obstacles = set()

        # Randomly place obstacles
        for _ in range(numObstacles):
            obstaclePosition = (np.random.randint(size), np.random.randint(size))
            self.obstacles.add(obstaclePosition)

    def reset(self):
        # Reseting the environment and return the starting position
        self.state = np.zeros((self.size, self.size))
        return self.startPosition

    def isValidPosition(self, position):
        # Checking if a position is valid within the grid
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def isObstacle(self, position):
        # Checking if a position is an obstacle
        return position in self.obstacles

    def isTerminal(self, position):
        # Checking if a position is the goal state
        return position == self.goalPosition

    def performAction(self, position, action):
        # Performing the specified action and return the new position and reward
        if action == 0:  # Up
            newPosition = (position[0] - 1, position[1])
        elif action == 1:  # Down
            newPosition = (position[0] + 1, position[1])
        elif action == 2:  # Left
            newPosition = (position[0], position[1] - 1)
        elif action == 3:  # Right
            newPosition = (position[0], position[1] + 1)
        else:  # Jump
            jumpDirection = np.random.choice([-1, 1], size=2)
            newPosition = (position[0] + jumpDirection[0], position[1] + jumpDirection[1])

        if self.isValidPosition(newPosition) and not self.isObstacle(newPosition):
            position = newPosition

        # Updating the state to mark the current position
        self.state = np.zeros((self.size, self.size))
        self.state[position] = 1  # Marking the current position in the state

        # Defining the reward based on the current state or other factors
        reward = -1 if not self.isTerminal(position) else 10

        return position, reward
