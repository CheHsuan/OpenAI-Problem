import numpy as np
import random
import sys

def init_Q_matrix():
    Q_matrix = np.ndarray(shape=(6, 6), dtype=int)
    Q_matrix.fill(0)
    return Q_matrix

def init_reward_matrix():
    reward_matrix = np.array([[-1, -1, -1, -1, 0, -1],
                     [-1, -1, -1, 0, -1, 100],
                     [-1, -1, -1, 0, -1,-1],
                     [-1, 0, 0, -1, 0, -1],
                     [0, -1, -1, 0, -1, 100],
                     [-1, 0, -1, -1, 0, 100]])
    return reward_matrix

def init_action_table(reward_matrix):
    action_table = dict();
    for i in range(6):
        actions = list()
        for j in range(6):
            if reward_matrix[i][j] >= 0:
                actions.append(j)
        action_table[i] = actions
    return action_table

def init(df=0):
    Q_matrix = init_Q_matrix()
    reward_matrix = init_reward_matrix()
    action_table = init_action_table(reward_matrix)
    return Q_matrix, reward_matrix, action_table

def Q(s, next_s, reward_matrix, Q_matrix, action_table, gamma):
    # Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
    Q_matrix[s][next_s] = reward_matrix[s][next_s] + gamma * max([Q_matrix[next_s][action_table[next_s][i]] for i in range(len(action_table[next_s]))])

def learning(Q_matrix, reward_matrix, action_table, episode, gamma=0, terminate_state=5):
    for i in range(episode):
        s = random.randint(0, 4)
        while not s == terminate_state:
            # select a random action a
            a = random.randint(0, len(action_table[s])-1)
            next_s = action_table[s][a]
            # calculate Q(s,a)
            Q(s, next_s, reward_matrix, Q_matrix, action_table, gamma)
            s = next_s
    print Q_matrix

def escape_building(Q_matrix, actoin_table, terminate_state=5):
    position = random.randint(0, 4)
    print "You are int the room %d" % position
    print "Now, start to escape from the building"
    route = list([position])
    while not position == terminate_state:
        maxQ = max(Q_matrix[position])
        candidate_action = np.intersect1d(np.argwhere(Q_matrix[position]==maxQ),action_table[position])
        if len(candidate_action) > 1:
            next_position = candidate_action[random.randint(0, len(candidate_action)-1)]
        else:
            next_position = candidate_action[0]
        route.append(next_position)
        position = next_position
    print "The route is ",
    print route

if __name__ == '__main__':
    Q_matrix, reward_matrix, action_table = init()
    if len(sys.argv) < 2:
        episode = 100
    else:
        episode = int(sys.argv[1])
    learning(Q_matrix, reward_matrix, action_table, episode, gamma=0.8)
    escape_building(Q_matrix, action_table)
