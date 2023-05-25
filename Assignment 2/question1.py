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
    # print(f'The next_state = (np.array(state) + action).tolist() outputs= \n {next_state} ')
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def optimal_policy(value):
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

def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    policy = np.random.randint(0,4,(5,5,1))
    
    epoch = 0
    while True:
        it = 0
        # it = 0
        while True:
            # keep iteration until convergence
            new_value = np.zeros_like(value)
            for i in range(WORLD_SIZE): #nested loop to loop over each state
                for j in range(WORLD_SIZE):
                    values = [] #values is different from value.
                    for x in policy[i, j]:
                        action = ACTIONS[x]
                        # print('The Action = ', action)
                   
                        (next_i, next_j), reward = step([i, j], action)
                        # value iteration
                        values.append(reward + DISCOUNT * value[next_i, next_j]) #V(s) = Sum over all s', r: p(Rt+1 + DISCOUNT * value[next_i, next_j])
                    new_value[i, j] = np.max(values)

            if np.sum(np.abs(new_value - value)) < 1e-2:
                np.set_printoptions(precision=2)
                print(f'The value function at epoch {epoch} coonverged at iteration {it} is \n {value} \n With the policy: \n {policy}')
                print()
                break
            value = new_value
            it += 1
            
            
             

        stable = True
        new_policy = optimal_policy(value)
        for i in range(len(new_policy)):
            for j in range(len(new_policy[i])):
                if not (new_policy[i, j].tolist() == policy[i,j].tolist()):
                    stable = False
        if stable == True:
            optimalPolicy = policy
            optimal_value =  value
            break
        
        # print(f'And the new_policy is in epoct {epoch}= \n{new_policy}')

        epoch += 1


        policy = new_policy
    return optimalPolicy, optimal_value
    # print(f'And the optimal policy is = \n{policy}')


policy, value = figure_3_5()
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
