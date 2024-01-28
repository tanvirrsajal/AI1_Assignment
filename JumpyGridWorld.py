# JumpyGridWorld.py
import numpy as np

class JumpyGridWorld:
    def __init__(self, size, numObstacles):
        # Initialize the grid world
        self.size = size
        self.state = np.zeros((size, size))
        self.startPosition = (0, 0)
        self.goalPosition = (size - 1, size - 1)
        self.obstacles = set()
        self.agentPath = []  # Store the agent's path during an episode

        # Randomly place obstacles
        for _ in range(numObstacles):
            obstaclePosition = (np.random.randint(size), np.random.randint(size))
            self.obstacles.add(obstaclePosition)

    def reset(self):
        # Reset the environment and return the starting position
        self.state = np.zeros((self.size, self.size))
        self.agentPath = []  # Clear the agent's path
        return self.startPosition

    def isValidPosition(self, position):
        # Check if a position is valid within the grid
        return 0 <= position[0] < self.size and 0 <= position[1] < self.size

    def isObstacle(self, position):
        # Check if a position is an obstacle
        return position in self.obstacles

    def isTerminal(self, position):
        # Check if a position is the goal state
        return position == self.goalPosition

    def performAction(self, position, action):
        # Perform the specified action and return the new position and reward
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

        self.state = np.zeros((self.size, self.size))
        self.state[position] = 1

        reward = -1 if not self.isTerminal(position) else 10

        # Add the current position to the agent's path
        self.agentPath.append(position)

        return position, reward
