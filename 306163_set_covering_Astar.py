from random import random
from functools import reduce
from collections import namedtuple
from queue import PriorityQueue
import numpy as np

# constant variables from problem settings: they can be modified to see different results
PROBLEM_SIZE = 5
SET_SIZE = 10
# create an array with random chioces between True/False with 20% probability that a variable is True
SETS = tuple(np.array([random() < .2 for _ in range(PROBLEM_SIZE)]) for _ in range(SET_SIZE))
State = namedtuple('State', ['taken', 'non_taken'])


# Define the class State() -> for each state we define the covered set, the remaining elements
# and the values of the following functions:
# g(n): actual cost so far to reach the node n
# h(n): heuristic function = estimated cost
class State:
    def __init__(self, covered_sets, remaining_elements, g_value, h_value, parent=None):
        self.covered_sets = covered_sets
        self.remaining_elements = remaining_elements
        self.g_value = g_value
        self.h_value = h_value
        self.parent = parent
        self.f_value = g_value + h_value
        # the f(n) function is given by the sum of g and h-> it is the total estimated cost of path through the node n

    def f_value(self):
        return self.g_value + self.h_value

    # functions to let the priority queue sort the new states basing of the comparison between integers (f values)
    def __gt__(self, other):
        return self.f_value > other.f_value

    def __lt__(self, other):
        return self.f_value < other.f_value


def covered(state):
    return reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )


def goal_check(state):
    return np.all(covered(state))


# define the universe of elements
universe = set(range(SET_SIZE))
# define the state
state = (set(range(SET_SIZE)), set)

# define the set of sets to cover the universe
# initialize sets_to_cover as an empty list
sets_to_cover = []

# represent the covered elements as a set
covered_elements = set()

# iterate through the SETS
for i, s in enumerate(SETS):
    # check if the s-th set covers any new elements in the universe
    new_elements = set(np.where(s)[0])

    # if they are covered, add this set to sets_to_cover
    if new_elements:
        sets_to_cover.append(i)

# define the goal state = all sets covered and expected distance form the solution equal to 0
goal_state = State(set(sets_to_cover), set(), 0, 0)

# from now on I follow the suggestion from https://brilliant.org/wiki/a-star-search/
#    make an openlist containing only the starting node
#    make an empty closed list
#    while (the destination node has not been reached):
#        consider the node with the lowest f score in the open list
#        if (this node is our destination node) :
#            we are finished
#        if not:
#            put the current node in the closed list and look at all of its neighbors
#            for (each neighbor of the current node):
#                if (neighbor has lower g value than current and is in the closed list) :
#                    replace the neighbor with the new, lower, g value
#                    current node is now the neighbor's parent
#                else if (current g value is lower and this neighbor is in the open list ) :
#                    replace the neighbor with the new, lower, g value
#                    change the neighbor's parent to our current node
#
#                else if this neighbor is not in both lists:
#                    add it to the open list and set its g


# Initialize the open list with the starting node
initial_state = State(set(), universe, 0, len(universe))
open_list = PriorityQueue() # which is our frontier
open_list.put((initial_state.f_value, initial_state))

closed_list = set()
counter = 0

while not open_list.empty():  # until I do not explore all
    counter += 1
    _, current_state = open_list.get()

    if current_state.covered_sets == goal_state.covered_sets:
        print("Solution Found:")
        print("Covered Sets:", current_state.covered_sets)
        print(f'Solved in {counter:,} steps')
        break

    if current_state in closed_list:
        continue

    closed_list.add(current_state)

    # add the uncovered set to the covered sets
    for s in sets_to_cover:
        if s not in current_state.covered_sets:
            new_covered_sets = current_state.covered_sets | {s}
            new_remaining_elements = current_state.remaining_elements - {s}
            new_g_value = current_state.g_value + 1

            # compute the new heuristic value h(n)
            new_h_value = len(new_remaining_elements)
            new_state = State(new_covered_sets, new_remaining_elements, new_g_value, new_h_value, parent=current_state)

            if new_state not in closed_list:
                open_list.put((new_state.f_value, new_state))

if current_state.covered_sets != goal_state.covered_sets:
    print("No solution found.")
