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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodDistances = [manhattanDistance(food, newPos) for food in newFood.asList()]
        ghostDistances = [manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates]
        if currentGameState.getPacmanPosition() == newPos or any(dist < 2 for dist in ghostDistances):
            return float("-inf")
        if len(foodDistances) == 0:
            return float("inf")
        scaredTimesSum = sum(newScaredTimes)
        return 1000 / sum(foodDistances) + 10000 / len(foodDistances) + (100 / scaredTimesSum if scaredTimesSum > 0 else 0)

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
        def max_value(gameState, depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            max_score = float("-inf")
            best_action = None
            for action in gameState.getLegalActions(0):
                score = min_value(gameState.generateSuccessor(0, action), 1, depth)
                if score[0] > max_score:
                    max_score = score[0]
                    best_action = action
            return max_score, best_action

        def min_value(gameState, agentID, depth):
            if len(gameState.getLegalActions(agentID)) == 0:
                return self.evaluationFunction(gameState), None
            min_score = float("inf")
            best_action = None
            for action in gameState.getLegalActions(agentID):
                next_state = gameState.generateSuccessor(agentID, action)
                if agentID == gameState.getNumAgents() - 1:
                    score = max_value(next_state, depth + 1)
                else:
                    score = min_value(next_state, agentID + 1, depth)
                if score[0] < min_score:
                    min_score = score[0]
                    best_action = action
            return min_score, best_action
        return max_value(gameState, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            max_score = float("-inf")
            best_action = None
            for action in gameState.getLegalActions(0):
                score = min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)
                if score[0] > max_score:
                    max_score = score[0]
                    best_action = action
                if max_score > beta:
                    return max_score, best_action
                alpha = max(alpha, max_score)
            return max_score, best_action
        
        def min_value(gameState, agentID, depth, alpha, beta):
            if len(gameState.getLegalActions(agentID)) == 0:
                return self.evaluationFunction(gameState), None
            min_score = float("inf")
            best_action = None
            for action in gameState.getLegalActions(agentID):
                next_state = gameState.generateSuccessor(agentID, action)
                if agentID == gameState.getNumAgents() - 1:
                    score = max_value(next_state, depth + 1, alpha, beta)
                else:
                    score = min_value(next_state, agentID + 1, depth, alpha, beta)
                if score[0] < min_score:
                    min_score = score[0]
                    best_action = action
                if min_score < alpha:
                    return min_score, best_action
                beta = min(beta, min_score)
            return min_score, best_action
        return max_value(gameState, 0, float("-inf"), float("inf"))[1]

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
        def max_value(gameState, depth):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            max_score = float("-inf")
            best_action = None
            for action in gameState.getLegalActions(0):
                score = expect_value(gameState.generateSuccessor(0, action), 1, depth)
                if score[0] > max_score:
                    max_score = score[0]
                    best_action = action
            return max_score, best_action
        
        def expect_value(gameState, agentID, depth):
            if len(gameState.getLegalActions(agentID)) == 0:
                return self.evaluationFunction(gameState), None
            total_score = 0
            best_action = None
            for action in gameState.getLegalActions(agentID):
                next_state = gameState.generateSuccessor(agentID, action)
                if agentID == gameState.getNumAgents() - 1:
                    score = max_value(next_state, depth + 1)
                else:
                    score = expect_value(next_state, agentID + 1, depth)
                total_score += score[0]
                best_action = action
            return total_score / len(gameState.getLegalActions(agentID)), best_action
        return max_value(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    foodDistances = [manhattanDistance(food, newPos) for food in newFood.asList()]
    ghostDistances = [manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates]
    score = currentGameState.getScore()
    score += sum([1.0 / distance for distance in foodDistances])
    # kept getting divide by zero errors so added 1e-1 to prevent that
    score -= sum([1.0 / (distance + 1e-1) for distance in ghostDistances])
    score -= len(newCapsules)
    return score
    

# Abbreviation
better = betterEvaluationFunction
