"""## Question 2

Starting from the code GridWorld 3 2.py, which is available on Canvas, implement the
complete value iteration algorithm to generate the optimal value function $v*$ and an optimal
policy \(\pi\)\(*\).
"""

def greedy_policy(value):
    policy = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=object)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            values = []
            for action_idx, action in enumerate(ACTIONS):
                (next_i, next_j), reward = step([i, j], action)
                values.append(reward + DISCOUNT * value[next_i, next_j])
            
            best_actions = np.argwhere(values == np.max(values)).flatten().tolist()
            policy[i, j] = np.array(best_actions)
    return policy

#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np


WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTION_PROB = 0.25

def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    it = 0
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                temp = []
                for action in ACTIONS:
                    
                    (next_i, next_j), reward = step([i, j], action)
                    temp.append(ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j]))
                    # bellman equation
                new_value[i, j] = max(temp) 
        if np.sum(np.abs(value - new_value)) < 1e-2:
            break
        value = new_value
        it += 1
        # input("Press Enter to continue...")
        np.set_printoptions(precision=2)
        print(value)
        print()
    print("Converges in {} iterations".format(it))
    

    ## Greedy Policy ##

    policy = greedy_policy(value)
    print("The Policy is",policy, sep = ' = n')

    return policy, value

if __name__ == '__main__':
    policy, value = figure_3_2()
    print(value)
    left, up, right, down = 0,1,2,3
    arrow_dic = dict([(0 , "left"), (1 ,"up"), (2 , "right"), (3, "down")])

    Arrow = np.zeros_like(policy, dtype='object')
    # print(Arrow)
    for i in range(len(policy)):
        for j in range(len(policy[i])):
            temp = []
            for index in range(len(policy[i,j])):
                x= arrow_dic[policy[i,j][index]]
                temp.append(x)
            Arrow[i,j] = temp    
    # print(Arrow.tolist())

    for i in range(len(policy)):
        for j in range(len(policy[i])):
            # for index in range(len(policy[i,j])):
            print(Arrow[i, j], end = ' , ')
        print()

#  (0 = left, 1 = up, 2 = right, 3 = down)
