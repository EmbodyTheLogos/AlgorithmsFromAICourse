########################################################
#
# CMPSC 441: Homework 3
#
########################################################

#References: A friend helped me debug my program.

student_name = 'Long Nguyen'
student_email = 'lhn5032@psu.edu'



########################################################
# Import
########################################################

from hw3_utils import *
from collections import deque
from queue import PriorityQueue
import math

# Add your imports here if used




##########################################################
# 1. Best-First, Uniform-Cost, A-Star Search Algorithms
##########################################################



def inQueue(child, frontier):
    for i in frontier.queue:
        if i[2] == child:
            return True
    return False

def getNode(child, frontier):
    for i in frontier.queue:
        if i[2] == child:
            return i


def best_first_search(problem):
    node = Node(problem.init_state, heuristic=problem.h(problem.init_state))
    frontier = PriorityQueue()         # queue: popleft/append-sorted
    explored = [problem.init_state]  # used as "visited"
    count = 0
    frontier.put((problem.h(node.state), count,  node))
    count = count +1
    while frontier:
        currentNode = frontier.get()[2]
        if problem.goal_test(currentNode.state):
            return currentNode
        else:
            for child in currentNode.expand(problem):
                if child.state not in explored and not inQueue(child, frontier):
                    frontier.put((problem.h(child.state), count, child))
                    count = count + 1
                    explored.append(child.state)
    return Node(None)


def uniform_cost_search(problem):
    node = Node(problem.init_state)
    frontier = PriorityQueue()       # queue: popleft/append-sorted
    explored = []                    # used as "expanded" (not "visited")

    count = 0
    frontier.put((node.path_cost, count, node))
    count = count + 1
    while frontier:
        currentNode = frontier.get()[2]
        if problem.goal_test(currentNode.state):
            return currentNode
        if currentNode in explored:
            continue
        explored.append(currentNode)
        for child in currentNode.expand(problem):
            if child not in explored:
                if not inQueue(child, frontier):
                    frontier.put((child.path_cost, count, child))
                    count = count + 1
                elif child.path_cost < getNode(child, frontier)[0]:
                    frontier.queue.remove(getNode(child, frontier))
                    frontier.put((child.path_cost, count, child))
                    count = count + 1
    return Node(None)



    
def a_star_search(problem):
    node = Node(problem.init_state, heuristic=problem.h(problem.init_state))
    frontier = PriorityQueue()       # queue: popleft/append-sorted
    explored = []                    # used as "expanded" (not "visited")

    count = 0
    frontier.put((node.path_cost + node.heuristic, count, node))
    count = count + 1
    while frontier:
        currentNode = frontier.get()[2]
        if problem.goal_test(currentNode.state):
            return currentNode
        if currentNode in explored:
            continue
        explored.append(currentNode)
        for child in currentNode.expand(problem):
            if child not in explored:
                if not inQueue(child, frontier):
                    frontier.put((child.path_cost + child.heuristic, count, child))
                    count = count + 1
                elif child.path_cost < getNode(child, frontier)[0]:
                    frontier.queue.remove(getNode(child, frontier))
                    frontier.put((child.path_cost + child.heuristic, count, child))
                    count = count + 1
    return Node(None)




##########################################################
# 2. N-Queens Problem
##########################################################


class NQueensProblem(Problem):
    """
    The implementation of the class NQueensProblem is given
    for those students who were not able to complete it in
    Homework 2.
    
    Note that you do not have to use this implementation.
    Instead, you can use your own implementation from
    Homework 2.

    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    """
    
    def __init__(self, n):
        super().__init__(tuple([-1] * n))
        self.n = n
        

    def actions(self, state):
        if state[-1] != -1:   # if all columns are filled
            return []         # then no valid actions exist
        
        valid_actions = list(range(self.n))
        col = state.index(-1) # index of leftmost unfilled column
        for row in range(self.n):
            for c, r in enumerate(state[:col]):
                if self.conflict(row, col, r, c) and row in valid_actions:
                    valid_actions.remove(row)
                    
        return valid_actions

        
    def result(self, state, action):
        col = state.index(-1) # leftmost empty column
        new = list(state[:])  
        new[col] = action     # queen's location on that column
        return tuple(new)

    
    def goal_test(self, state):
        if state[-1] == -1:   # if there is an empty column
            return False;     # then, state is not a goal state

        for c1, r1 in enumerate(state):
            for c2, r2 in enumerate(state):
                if (r1, c1) != (r2, c2) and self.conflict(r1, c1, r2, c2):
                    return False
        return True

    
    def conflict(self, row1, col1, row2, col2):
        return row1 == row2 or col1 == col2 or abs(row1-row2) == abs(col1-col2)

    
    def g(self, cost, from_state, action, to_state):
        """
        Return path cost from start state to to_state via from_state.
        The path from start_state to from_state costs the given cost
        and the action that leads from from_state to to_state
        costs 1.
        """
        return cost + 1



    def h(self, state):
        """
        Returns the heuristic value for the given state.
        Use the total number of conflicts in the given
        state as a heuristic value for the state.
        """
        count = 0
        for c1, r1 in enumerate(state):
            for c2, r2 in enumerate(state):
                if (r1, c1) != (r2, c2) and self.conflict(r1, c1, r2, c2):
                    count = count + 1
        return count



##########################################################
# 3. Graph Problem
##########################################################



class GraphProblem(Problem):
    """
    The implementation of the class GraphProblem is given
    for those students who were not able to complete it in
    Homework 2.
    
    Note that you do not have to use this implementation.
    Instead, you can use your own implementation from
    Homework 2.

    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    >>>> USE THIS IMPLEMENTATION AT YOUR OWN RISK <<<<
    """

    def __init__(self, init_state, goal_state, graph):
        super().__init__(init_state, goal_state)
        self.graph = graph

    def actions(self, state):
        """Returns the list of adjacent states from the given state."""
        return list(self.graph.get(state).keys())

    def result(self, state, action):
        """Returns the resulting state by taking the given action.
            (action is the adjacent state to move to from the given state)"""
        return action

    def goal_test(self, state):
        return state == self.goal_state

    def g(self, cost, from_state, action, to_state):
        """
        Returns the path cost from root to to_state.
        Note that the path cost from the root to from_state
        is the give cost and the given action taken at from_state
        will lead you to to_state with the cost associated with
        the action.
        """

        adjacentCity = self.graph.edges[from_state]
        costToAdjacent = adjacentCity[to_state]
        return cost + costToAdjacent
    

    def h(self, state):
        """
        Returns the heuristic value for the given state. Heuristic
        value of the state is calculated as follows:
        1. if an attribute called 'heuristics' exists:
           - heuristics must be a dictionary of states as keys
             and corresponding heuristic values as values
           - so, return the heuristic value for the given state
        2. else if an attribute called 'locations' exists:
           - locations must be a dictionary of states as keys
             and corresponding GPS coordinates (x, y) as values
           - so, calculate and return the straight-line distance
             (or Euclidean norm) from the given state to the goal
             state
        3. else
           - cannot find nor calculate heuristic value for given state
           - so, just return a large value (i.e., infinity)
        """
        if hasattr(self.graph, 'heuristics'):
            return self.graph.heuristics[state]
        elif hasattr(self.graph, 'locations'):
            cityPosition = self.graph.locations[state]
            goalPosition = self.graph.locations[self.goal_state]
            distance = ((goalPosition[0] - cityPosition[0]) ** 2) + ((goalPosition[1] - cityPosition[1]) ** 2)
            distance = math.sqrt(distance)
            return distance
        else:
            return math.inf



##########################################################
# 4. Eight Puzzle
##########################################################


class EightPuzzle(Problem):
    def __init__(self, init_state, goal_state=(1,2,3,4,5,6,7,8,0)):
        super().__init__(init_state, goal_state)
    

    def actions(self, state):
        lst = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        if state[0] == 0:
            lst.remove('UP')
            lst.remove('LEFT')
        elif state[1] == 0:
            lst.remove('UP')
        elif state[2] == 0:
            lst.remove('UP')
            lst.remove('RIGHT')
        elif state[3] == 0:
            lst.remove('LEFT')
        elif state[5] == 0:
            lst.remove('RIGHT')
        elif state[6] == 0:
            lst.remove('LEFT')
            lst.remove('DOWN')
        elif state[7] == 0:
            lst.remove('DOWN')
        elif state[8] == 0:
            lst.remove('RIGHT')
            lst.remove('DOWN')
        return lst

    
    def result(self, state, action):
        lst = list(state)
        index = -1
        for i in lst:
            index=index+1
            if(i == 0):
                break
        if action in self.actions(state):
            if action == 'RIGHT':
                temp = lst[index]
                lst[index] = lst[index+1]
                lst[index+1] = temp
            elif action == 'LEFT':
                temp = lst[index]
                lst[index] = lst[index - 1]
                lst[index - 1] = temp
            elif action == 'UP':
                temp = lst[index]
                lst[index] = lst[index - 3]
                lst[index - 3] = temp
            elif action == 'DOWN':
                temp = lst[index]
                lst[index] = lst[index + 3]
                lst[index + 3] = temp
            return tuple(lst)
        else:
            return state

    
    def goal_test(self, state):
        return state == self.goal_state
    

    def g(self, cost, from_state, action, to_state):
        """
        Return path cost from root to to_state via from_state.
        The path from root to from_state costs the given cost
        and the action that leads from from_state to to_state
        costs 1.
        """
        return cost + 1
    

    def h(self, state):
        """
        Returns the heuristic value for the given state.
        Use the sum of the Manhattan distances of misplaced
        tiles to their final positions.
        """
        x = -1
        y = -1
        board = {}
        for i in range(9):
            y = y + 1
            if i % 3 == 0:
                x= x + 1
                y=0
            board[state[i]] = (x, y)

        x = -1
        y = -1
        boardGoal = {}
        for i in range(9):
            y = y + 1
            if i % 3 == 0:
                x= x + 1
                y=0
            boardGoal[self.goal_state[i]] = (x, y)
        mahantan = 0
        for i in range(1, 9):
            point = board[i]
            pointGoal = boardGoal[i]
            mahantan = mahantan + abs(pointGoal[0] - point[0])
            mahantan = mahantan + abs(pointGoal[1] - point[1])

        return mahantan
