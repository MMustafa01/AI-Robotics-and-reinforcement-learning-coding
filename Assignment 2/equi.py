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
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

def optimal_policy(value):
    policy = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=int)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            values = []
            for action_idx, action in enumerate(ACTIONS):
                (next_i, next_j), reward = step([i, j], action)
                values.append(reward + DISCOUNT * value[next_i, next_j])
            best_actions = np.argwhere(values == np.max(values)).flatten().tolist()
            policy[i, j] = np.random.choice(best_actions)
    return policy

def main():
    value = np.array([[3.31, 8.9, 4.0, 5.89, 1.55],
                      [1.48, 2.92, 2.54, 2.45, 0.51],
                      [0.07, 0.65, 0.85, 0.43, -0.38],
                      [-0.78, -0.36, -0.23, -0.44, -1.11],
                      [-1.97, -1.36, -1.23, -1.43, -2.1]])

    optimal_policy_array = optimal_policy(value)
    print(optimal_policy_array)

if __name__ == '__main__':
    main()
