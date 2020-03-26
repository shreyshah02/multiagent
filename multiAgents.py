# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # print("The successor game state: ", successorGameState)
        # print("The new position: ", newPos)
        # print("new food: ", newFood)
        # print("New Ghost states: ", newGhostStates)
        # print("New Scared times: ", newScaredTimes)

        # Getting the remaining food in the grid as a list
        Food_List = newFood.asList()

        # dist variable is used to get the minimum distance from pacman's new position to a food pellet
        # Initially dist is set to infinity
        dist = float('inf')

        # Determining the distance between pacman's new position and the remaining food pellets using for loop
        # and updating dist to have minimum distance
        for food in Food_List:
            x = abs(food[0] - newPos[0]) + abs(food[1] - newPos[1])
            dist = min(dist, x)
            #dist += manhattanDistance(food, newPos)

        # Taking the reciprocal of the dist if it is greater than 0
        # Taking the reciprocal since we want the state with lesser distance to have a better evaluation
        if dist > 0:
            dist = 1/dist

        #Getting the positions of all the ghosts in the current game state
        ghosts = currentGameState.getGhostPositions()

        # ghost_dist to keep the minimum distance between the new pacman position and the ghosts
        # Initially set to infinity
        ghost_dist = float('inf')
        # variable i to keep the count of current ghost to index the ghost's scared time
        i = 0

        # Looping over the ghosts to determine the minimum distance between the pacman and the ghosts
        for ghost in ghosts:
            # If the current ghost is scared then we don't consider it, thus continue
            if newScaredTimes[i]:
                # Incrementing the counter
                i += 1
                continue
            g_dist = manhattanDistance(ghost, newPos)
            ghost_dist = min(ghost_dist, g_dist)
            # Incrementing the counter
            i += 1

        # If the minimum ghost distance is greater than 5 then we set it to 5
        # Otherwise the pacman goes to the wall farthest from the ghost and stays there
        if ghost_dist > 5:
            ghost_dist = 5
        #return dist + ghost_dist

        # returning the sum of score + dist + ghost_dist as the evaluation for a successor state
        return successorGameState.getScore() + dist + ghost_dist



def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        #util.raiseNotDefined()

        # Setting the starting agent to the index variable, which is 0. Hence the initial agent is pacman
        agent = self.index

        # Getting the number of agents. Defining it as an attribute since we need to access it in all methods
        self.numAgents = gameState.getNumAgents()

        # Getting the required depth for the search. d monitors the remaining required depth levels
        d = self.depth

        # Calling the Class Method MiniMax to get the minimax value of the correct action for pacman
        minimaxVal, action = self.MiniMax(gameState, agent, d-1)

        # Returning the action to be taken by the agent according to minimax algorithm
        return action

    def MiniMax(self, state, agent, depth):
        """ The class implementing the Minimax Algorithm. Returns the minimax value for the current state and the
        optimal action to be taken"""

        # Getting the legal actions for the agent according to game rules
        actions = state.getLegalActions(agent)

        # Initializing act as an empty list. act will keep track of the optimal action at current state for the agent
        act =[]

        # If we don't have any legal actions i.e. we have reached a terminal state or if we have completed the depth
        # requirement then we evaluate the current state and return the evaluated value along an empty act
        if len(actions) == 0 or depth < 0:
            # evaluating the current state
            minimaxVal = self.evaluationFunction(state)
            return minimaxVal, act

        # Determining the index for the next agent
        # If the current agent is the last ghost then the next agent should be pacman, i.e. value 0 and decrement the
        # remaining depth levels as we will enter the next depth level
        # else we increment the agent value to get the next agent
        if agent == self.numAgents - 1:
            nextAgent = 0
            depth -= 1
        else:
            nextAgent = agent + 1

        # If the current agent is the pacman, we choose an action corresponding to the maximum minimax value
        if agent == 0:
            #depth-=1

            # Initially setting the minimax value to negative infinity
            minimaxVal = float('-inf')

            # Looping over all the legal actions
            for action in actions:
                # Getting the new gamestate state achieved by implementing the current action for the current agent
                newState = state.generateSuccessor(agent, action)

                # Determinign the minimax value of the next state by reccursive call to the MiniMax method with the
                # new state, updated depth and the next agent
                v = self.MiniMax(newState, nextAgent, depth)[0]

                # If the minimax value of the next state is greater than the current minimax value, we update the
                # minimax value with this value and store the action corresponding to this minimax value in act
                if v > minimaxVal:
                    minimaxVal = v
                    act = action
            # return minimaxVal,[act]

        # If the agent is a ghost, we choose an action corresponding to the minimum minimax value
        else:
            # Setting the minimax value to be infinity initially
            minimaxVal = float('inf')

            # Looping over the legal actions
            for action in actions:
                # Getting the new gamestate state achieved by implementing the current action for the current agent
                newState = state.generateSuccessor(agent, action)

                # Determinign the minimax value of the next state by reccursive call to the MiniMax method with the
                # new state, updated depth and the next agent
                v = self.MiniMax(newState, nextAgent, depth)[0]

                # If the minimax value of the next state is smaler than the current minimax value, we update the
                # minimax value with this value and store the action corresponding to this minimax value in act
                if v < minimaxVal:
                    minimaxVal = v
                    act = action

        # Returning the minimax value for the current state as well as the corresponding action
        return minimaxVal, act

    # First attempt at implementing the minimax algorithm

    #     if gameState.isWin() or gameState.isLose():
    #         return self.evaluationFunction(gameState)
    #     #agent = 0 # Pacman
    #     actions = gameState.getLegalActions(agent)
    #     if len(actions) == 0:
    #         return []
    #     MiniMaxValue = []
    #     for action in actions:
    #         newState = gameState.generateSuccessor(agent, action)
    #         if d>0:
    #             MiniMaxValue.append(self.Minivalue(newState, agent+1, d-1))
    #         else :
    #             MiniMaxValue.append(self.evaluationFunction(newState))
    #     return actions[MiniMaxValue.index(max(MiniMaxValue))]
    #
    # def Minivalue(self, state, agent, d):
    #     if state.isWin() or state.isLose():
    #         return self.evaluationFunction(state)
    #     v = float('inf')
    #     actions = state.getLegalActions(agent)
    #     if len(actions) == 0:
    #         v = self.evaluationFunction(state)
    #         return v, []
    #     for action in actions:
    #         newState = state.generateSuccessor(agent, action)
    #         if d == 0 and agent == self.numAgents - 1:
    #             v = min(v, self.evaluationFunction(newState))
    #
    #         elif agent == self.numAgents - 1:
    #             agent = 0
    #             v = min(v, self.Maxvalue(newState, agent, d))
    #         else:
    #             v = min(v, self.Minivalue(newState, agent+1, d))
    #     return v
    #
    # def Maxvalue(self, state, agent, d):
    #     if state.isWin() or state.isLose():
    #         return self.evaluationFunction(state)
    #     d = d-1
    #     v = float('-inf')
    #     actions = state.getLegalActions(agent)
    #     if len(actions) == 0:
    #         v = self.evaluationFunction(state)
    #         return v
    #     for action in actions:
    #         newState = state.generateSuccessor(agent, action)
    #         v = max(v, self.Minivalue(newState, agent+1, d))
    #     return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # Getting the required depth for the search. d monitors the remaining required depth levels
        d = self.depth
        # Setting the starting agent to the index variable, which is 0. Hence the initial agent is pacman
        agent = self.index
        # Getting the number of agents. Defining it as an attribute since we need to access it in all methods
        self.numAgents = gameState.getNumAgents()
        # Initializing alpha and beta
        alpha = float('-inf')
        beta = float('inf')
        # Method 1
        # Calling the Class Method AlphaBeta to get the minimax value of the correct action for pacman
        value, action = self.AlphaBeta(gameState, alpha, beta, d-1, agent)
        # Method 2
        # Comment Method 1, uncomment method 2 and use the next statement
        # value, action = self.MaxValue(gameState, alpha, beta, d - 1, agent)
        return action

    def AlphaBeta(self, state, alpha, beta, depth, agent):
        # Getting the legal actions for the agent according to game rules
        # Initializing act as an empty list. act will keep track of the optimal action at current state for the agent
        act = []
        actions = state.getLegalActions(agent)

        # If we don't have any legal actions i.e. we have reached a terminal state or if we have completed the depth
        # requirement then we evaluate the current state and return the evaluated value along an empty act
        if len(actions) == 0 or depth<0:
            return self.evaluationFunction(state), []

        # Determining the index for the next agent
        # If the current agent is the last ghost then the next agent should be pacman, i.e. value 0 and decrement the
        # remaining depth levels as we will enter the next depth level
        # else we increment the agent value to get the next agent
        if agent == self.numAgents - 1:
            nextAgent = 0
            depth-=1
        else:
            nextAgent = agent+1

        # Checkpoint Method 1: 1st implementation. This also works.

        # if agent == 0:
        #     #depth-=1
        #     minimaxVal = float('-inf')
        #     for action in actions:
        #         newState = state.generateSuccessor(agent, action)
        #         v = self.AlphaBeta(newState, alpha, beta, depth, nextAgent)[0]
        #         if v>minimaxVal:
        #             minimaxVal = v
        #             act = action
        #         if minimaxVal > beta:
        #             return minimaxVal, act
        #         if minimaxVal > alpha:
        #             alpha = minimaxVal
        #     # return minimaxVal,act
        # else:
        #     minimaxVal = float('inf')
        #     for action in actions:
        #         newState = state.generateSuccessor(agent, action)
        #         # if agent == self.numAgents-1:
        #         #     if depth == 0:
        #         #         v = self.evaluationFunction(newState)
        #         #     else:
        #         v = self.AlphaBeta(newState, alpha, beta, depth, nextAgent)[0]
        #         if v<minimaxVal:
        #             minimaxVal=v
        #             act = action
        #         if minimaxVal < alpha:
        #             return minimaxVal, act
        #         if minimaxVal<beta:
        #             beta = minimaxVal
        # return minimaxVal, act

        # CheckPoint: 2nd implementation. Similar to the 1st implementation, just cleaner

        # If the current agent is Pacman, minimax value is set to negative infinity initially
        if agent == 0:
            value = float('-inf')
        # else it is set to infinity i.e. for ghosts
        else:
            value = float('inf')

        # Looping over all the legal actions
        for action in actions:

            # Getting the new game state for the current action and agent
            newState = state.generateSuccessor(agent, action)
            # Determining the minimax value for the new state with updated depth, alpha, beta, and new agent
            v = self.AlphaBeta(newState, alpha, beta, depth, nextAgent)[0]

            # For pacman agent
            if agent == 0:
                # Updating the minimax value with the max value between v and value and storing that action in act
                if v > value:
                    value = v
                    act = action
                # Implementing the alpha beta pruning
                if value > beta:
                    return value, act
                if value > alpha:
                    alpha = value
            # For ghost
            else:
                # Updating the minimax value with the min value between v and value and storing that action in act
                if v < value:
                    value = v
                    act = action
                # Implementing the alpha beta pruning for the ghost agent
                if value < alpha:
                    return value, act
                if value < beta:
                    beta = value
        # Returning the minimax value and the corresponding action for the current state
        return value, act

# Method 2: this also works

    # def MaxValue(self, state, alpha, beta, depth, agent):
    #     actions = state.getLegalActions(agent)
    #     if len(actions) == 0 or depth<0:
    #         value = self.evaluationFunction(state)
    #         return value, []
    #     act = []
    #     value = float('-inf')
    #
    #     for action in actions:
    #         newState = state.generateSuccessor(agent, action)
    #         v = self.MinValue(newState, alpha, beta, depth, agent+1)[0]
    #         if v>value:
    #             value = v
    #             act = action
    #         if value>beta:
    #             return value, act
    #         if value>alpha:
    #             alpha = value
    #     return value, act
    # def MinValue(self, state, alpha, beta, depth, agent):
    #     actions = state.getLegalActions(agent)
    #     if len(actions) == 0 or depth<0:
    #         value = self.evaluationFunction(state)
    #         return value, []
    #     act = []
    #     value = float('inf')
    #
    #     if agent == self.numAgents - 1:
    #         NextAgent = 0
    #         depth -=1
    #     else:
    #         NextAgent = agent+1
    #     for action in actions:
    #         newState = state.generateSuccessor(agent, action)
    #
    #         if agent == self.numAgents - 1:
    #             # if depth == 0:
    #             #     v = self.evaluationFunction(newState)
    #             # else:
    #
    #             v = self.MaxValue(newState, alpha, beta, depth, NextAgent)[0]
    #         else:
    #             v = self.MinValue(newState, alpha, beta, depth, NextAgent)[0]
    #         if v<value:
    #             value = v
    #             act = action
    #         if value<alpha:
    #             return value, act
    #         if value<beta:
    #             beta = value
    #     return value, act


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # Setting the starting agent to the index variable, which is 0. Hence the initial agent is pacman
        agent = self.index
        # Getting the required depth for the search. d monitors the remaining required depth levels
        d = self.depth
        # Getting the number of agents. Defining it as an attribute since we need to access it in all methods
        self.numAgents = gameState.getNumAgents()

        _, action = self.Expectimax(gameState, agent, d - 1)
        return action

    def Expectimax(self, state, agent, depth):
        # Getting the legal actions for the agent
        actions = state.getLegalActions(agent)

        # If we don't have any legal actions i.e. we have reached a terminal state or if we have completed the depth
        # requirement then we evaluate the current state and return the evaluated value along an empty act
        if len(actions) == 0 or depth < 0:
            return self.evaluationFunction(state), []
        # Initializing act as an empty list. act will keep track of the optimal action at current state for the agent
        act = []

        # Setting the probability for each action = 1/number of actions for the ghost, since the probability is uniform
        prob = 1/len(actions)

        # Determining the index for the next agent
        # If the current agent is the last ghost then the next agent should be pacman, i.e. value 0 and decrement the
        # remaining depth levels as we will enter the next depth level
        # else we increment the agent value to get the next agent
        if agent == self.numAgents - 1:
            nextAgent = 0
            depth -= 1
        else:
            nextAgent = agent + 1

        # If agent is pacman, initialize the value with negative infinity, else initialize it to 0
        if agent == 0:
            value = float('-inf')
        else:
            value = 0

    # Looping over all legal actions
        for action in actions:
            # Get the new state for the current agent and action
            newState = state.generateSuccessor(agent, action)

            # Get the expectimax value for the newstate with the next agent and the updated depth
            v = self.Expectimax(newState, nextAgent, depth)[0]

            # If the agent is pacman
            if agent == 0:
                # Updating the minimax value with the max value between v and value and storing that action in act
                if v > value:
                    value = v
                    act = action
            # if the agent is ghost
            else:
                # the value is the probability weighted average of the expectimax value
                value += prob*v
        # Return the expectimax value for the current state along with the corresponding action
        return value, act


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>

    For this Evaluation function, a state is evaluated based on the score of the current state, the distance to the
    nearest food pellet, and the distance to the nearest non-scared ghost. Since for distance to food, the lesser the
    better, I have taken the reciprocal of the distance. The distance to the nearest ghost is thresholded at value '5'
    i.e. if the dist to ghost > 5, it is made equal to 5. Otherwise the agent will disregard the food go to the wall
    farthest from the ghosts and stay there
    The final evaluation is the sum of all these quantities i.e. :
    score of the current state, reciprocal of the distance to the nearest food pellet, and the min distance to the ghost

    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    pacmanPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Capsules = currentGameState.getCapsules()

    Food_List = Food.asList()
    dist = float('inf')
    capDist = float('inf')
    for food in Food_List:
        x = abs(food[0] - pacmanPos[0]) + abs(food[1] - pacmanPos[1])
        dist = min(dist, x)
        # dist += manhattanDistance(food, newPos)
    if dist > 0:
        dist = 1 / dist
    # for cap in Capsules:
    #     x = manhattanDistance(pacmanPos, cap)
    #     capDist = min(x, capDist)
    # if capDist > 0:
    #     capDist = 1/capDist
    ghosts = currentGameState.getGhostPositions()
    ghost_dist = float('inf')
    i = 0
    for ghost in ghosts:
        if ScaredTimes[i]:
            i += 1
            continue
            # g_dist = - ScaredTimes[i]*manhattanDistance(ghost, pacmanPos)
        else:
            g_dist = manhattanDistance(ghost, pacmanPos)
        ghost_dist = min(ghost_dist, g_dist)
        i += 1
    if ghost_dist > 5:
        ghost_dist = 5
    # return dist + ghost_dist
    return currentGameState.getScore() + dist + ghost_dist #+ capDist




# Abbreviation
better = betterEvaluationFunction
