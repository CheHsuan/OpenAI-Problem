import gym
import numpy as np
import tensorflow as tf
import sys

# define the policy network - a 3-layer nerual network
def policy_network():
    input_dim  = 4
    hidden_dim = 10
    output_dim = 2
    with tf.variable_scope("policy"):
        # Define input and output dimension.
        tf_observation = tf.placeholder(tf.float32, shape=(None, input_dim))
        tf_r_probability = tf.placeholder(tf.float32, shape=(None, output_dim))
        tf_advantage = tf.placeholder(tf.float32,[None,1])

        # Variables.
        weights1 = tf.Variable(tf.truncated_normal([input_dim, hidden_dim]))
        biases1 = tf.Variable(tf.zeros([hidden_dim]))
        weights2 = tf.Variable(tf.truncated_normal([hidden_dim, output_dim]))
        biases2 = tf.Variable(tf.zeros([output_dim]))

        # Training computation.
        hidden_layer = tf.nn.relu(tf.matmul(tf_observation, weights1) + biases1)
        output_layer = tf.matmul(hidden_layer, weights2) + biases2
        p_probability = tf.nn.softmax(output_layer)

        # Optimizer.
        good_probabilities = tf.reduce_sum(tf.mul(p_probability, tf_r_probability),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * tf_advantage
        loss = -tf.reduce_sum(eligibility)
        policy_optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        return tf_observation, tf_r_probability, tf_advantage, p_probability, policy_optimizer

# define the value network - regression
def value_network():
    input_dim  = 4
    hidden_dim = 10
    output_dim = 1
    with tf.variable_scope("value"):
        # Define input and output dimension.
        tf_state = tf.placeholder(tf.float32, [None, input_dim])
        tf_r_value = tf.placeholder(tf.float32, [None, output_dim])

        # Variables.
        weights1 = tf.Variable(tf.truncated_normal([input_dim, hidden_dim]))
        biases1 = tf.Variable(tf.zeros([hidden_dim]))
        weights2 = tf.Variable(tf.truncated_normal([hidden_dim, output_dim]))
        biases2 = tf.Variable(tf.zeros([output_dim]))

        # Training computation.
        hidden_layer = tf.nn.relu(tf.matmul(tf_state, weights1) + biases1)
        p_value = tf.matmul(hidden_layer, weights2) + biases2

        # Training computation.
        loss = tf.nn.l2_loss(p_value - tf_r_value)

        # Optimizer.
        value_optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return tf_state, tf_r_value, p_value, value_optimizer

def onehot_encoding(target, dim=2):
    onehot = np.zeros(dim)
    onehot[np.argmax(target)] = 1
    return onehot

def choose_action(probability, dim=2):
    action = np.random.choice(dim, 1, p=probability[0])[0]
    actionblank = np.zeros(2)
    actionblank[action] = 1
    return actionblank, action

def run_episode(env, policy_net, value_net, sess):
    tf_observation, tf_r_probability, tf_advantage, p_probability, policy_optimizer = policy_net
    tf_state, tf_r_value, p_value, value_optimizer = value_net
    # reset the env
    observation = env.reset()
    total_reward = 0
    obs_record = list()
    act_record = list()
    for t in xrange(200):
        probability = p_probability.eval(feed_dict={tf_observation : [observation]})
        actionblank, action = choose_action(probability)
        obs_record.append(observation)
        observation, reward, done, info = env.step(action)        
        act_record.append(actionblank)
        total_reward += reward
        if done:
            print "Episode finished after %d timesteps" % (t+1)
            break

    # training
    transitions = list()
    state_values = list()
    for i in range(len(obs_record)):
        state_value = 0
        decrease = 1
        for j in range(len(obs_record)-i):
            state_value += decrease
            decrease = decrease * 0.97
        state_values.append(state_value)
        future_value = state_value
        current_value = p_value.eval(feed_dict={tf_state : [obs_record[i]]})[0][0]

        transition = future_value - current_value
        transitions.append(transition)

    # train value network 
    state_value_vec = np.expand_dims(state_values, axis=1)
    sess.run(value_optimizer,feed_dict={tf_state : obs_record, tf_r_value : state_value_vec})
    # train policy network
    transition_vec = np.expand_dims(transitions, axis=1)
    feed_dict = {tf_observation : obs_record, tf_r_probability : act_record, tf_advantage : transition_vec}
    sess.run(policy_optimizer, feed_dict=feed_dict)

    return total_reward 

def main():
    # define the game environment - carpole
    env = gym.make('CartPole-v0')
    p_net = policy_network()
    v_net = value_network()
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    rewards = list()
    for i_episode in range(int(sys.argv[1])):
        reward = run_episode(env, p_net, v_net, sess)
        if reward > 199:
            print "Achieve 200 at %d episode" % i_episode
            break
        rewards.append(reward)
    print "Average timesteps : %d" % np.mean(rewards)
    print "Variance : %d" % np.var(rewards)

if __name__ == "__main__":
    main()
