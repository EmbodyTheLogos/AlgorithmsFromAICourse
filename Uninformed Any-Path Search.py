########################################################
#
# CMPSC 441: Homework 2
#
########################################################


student_name = 'Long Nguyen'
student_email = 'lhn5032@psu.edu'




########################################################
# Import
########################################################


from hw2_utils import *
from collections import deque





##########################################################
# 1. Uninformed Any-Path Search Algorithms
##########################################################


def depth_first_search(problem):
    
    node = Node(problem.init_state)
    frontier = deque([node])         # stack: append/pop
    explored = [problem.init_state]  # used as "visited"

    while frontier:
        currentNode = frontier.pop()
        if problem.goal_test(currentNode.state):
            return currentNode
        else:
            for child in currentNode.expand(problem):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                    explored.append(child.state)
    currentNode.state = None
    return currentNode


def breadth_first_search(problem):
    
    node = Node(problem.init_state)
    frontier = deque([node])         # stack: append/popleft
    explored = [problem.init_state]  # used as "visited"

    while frontier:
        currentNode = frontier.popleft()
        if problem.goal_test(currentNode.state):
            return currentNode
        else:
            for child in currentNode.expand(problem):
                if child.state not in explored and child not in frontier:
                    frontier.append(child)
                    explored.append(child.state)
    currentNode.state = None
    return currentNode



##########################################################
# 2. N-Queens Problem
##########################################################


class NQueensProblem(Problem):
    
    def __init__(self, n):
        lst = []
        for i in range(n):
            lst.append(-1)
        Problem(tuple(lst))
        self.init_state = tuple(lst)
        self.n = n
    
    def actions(self, state):
        lst = []
        for i in range(len(state)):
            lst.append(i)
        for i in range(len(state)):
            if state[i] == -1:
                index = i
                break

        ###check for existing and diagonal rows:
        for i in range(index):
            upperDiagonal = state[i] - abs(index - i)
            lowerDiagonal = state[i] + abs(index - i)
            if state[i] in lst:
                lst.remove(state[i])
            if upperDiagonal >= 0 and upperDiagonal in lst:
                lst.remove(upperDiagonal)
            if lowerDiagonal <= len(state) and lowerDiagonal in lst:
                lst.remove(lowerDiagonal)
        return lst


    def result(self, state, action):
        lst = self.actions(state)
        listState = list(state)
        for i in range(len(listState)):
            if listState[i] == -1:
                index = i
                break
        if action in lst:
            listState[index] = action
        state = tuple(listState)
        return state
    
                        
    def goal_test(self, state):
        currentState = self.init_state
        for i in range(len(state)):
            currentState = self.result(currentState, state[i])

        return -1 not in currentState

##########################################################
# 3. Farmer's Problem
##########################################################


class FarmerProblem(Problem):
    
    def __init__(self, init_state, goal_state):
        Problem(init_state, goal_state)
        self.init_state = init_state
        self.goal_state = goal_state

    
    def actions(self, state):
        lst = ['F', 'FG', 'FC', 'FX']
        if state[1] == state[2] or state[2] == state[3]:
            lst.remove('F')
        if state[1] == state[2] or state[0] != state[3]:
            lst.remove('FX')
        if state[2] == state[3] or state[0] != state[1]:
            lst.remove('FG')
        if state[0] != state[2]:
            lst.remove('FC')
        return lst
    
    def result(self, state, action):
        lst = self.actions(state)
        state = list(state)
        if action in lst:
            if action == 'F':
                state[0] = not state[0]
            if action == 'FG':
                state[0] = not state[0]
                state[1] = not state[1]
            if action == 'FC':
                state[0] = not state[0]
                state[2] = not state[2]
            if action == 'FX':
                state[0] = not state[0]
                state[3] = not state[3]
        return tuple(state)
    
    def goal_test(self, state):
        return state == self.goal_state



##########################################################
# 4. Graph Problem
##########################################################


class GraphProblem(Problem):
    
    def __init__(self, init_state, goal_state, graph):
        Problem(init_state, goal_state)
        self.init_state = init_state
        self.goal_state = goal_state
        self.graph = graph

    
    def actions(self, state):
        lst = [keys for keys in self.graph.edges[state]]
        return lst

    
    def result(self, state, action):
        lst = self.actions(state)
        if action in lst:
            return action
        else:
            return state

    def goal_test(self, state):
        return state == self.goal_state