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

def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    # print('The next_state = (np.array(state) + action).tolist() outputs= \n {next_state} ')
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    policy = [(0,1,2,3), (0,1,2,3), (0,1,2,3),(0,1,2,3), (0,1,2,3)]
    policy = [policy, policy, policy, policy, policy]
    
    # raise('error')
    it = 0
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        
        for i in range(WORLD_SIZE): #nested loop to loop over each state
            for j in range(WORLD_SIZE):
                values = [] #values is different from value.
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j]) #V(s) = Sum over all s', r: p(Rt+1 + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-2:
            break
        value = new_value
        it += 1
        if it == 20:
            break
        # input("Press Enter to continue...")
        np.set_printoptions(precision=2)
        print(f'The value function at iteration {it} is \n {value}')
        print()
    print("Converges in {} iterations".format(it))


###### Policy improvement code goes here





if __name__ == '__main__':
    figure_3_5()
