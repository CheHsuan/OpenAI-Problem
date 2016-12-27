# -*- coding: utf-8 -*-
import sys
import gym
import numpy as np
import tensorflow as tf

def policy_network():
    input_dim  = 80 * 80
    hidden_dim = 200 
    output_dim = 2
    with tf.variable_scope("policy"):
        # Define input and output dimension.
        observation = tf.placeholder(tf.float32, shape=(None, input_dim))
        action = tf.placeholder(tf.float32, shape=(None, output_dim))
        reward = tf.placeholder(tf.float32)

        # Variables.
        w1 = tf.Variable(tf.truncated_normal([input_dim, hidden_dim]))
        b1 = tf.Variable(tf.zeros([hidden_dim]))
        w2 = tf.Variable(tf.truncated_normal([hidden_dim, output_dim]))
        b2 = tf.Variable(tf.zeros([output_dim]))   

        # Training computation.
        hidden_layer = tf.nn.relu(tf.matmul(observation, w1) + b1)
        output_layer = tf.matmul(hidden_layer, w2) + b2
        probability = tf.nn.softmax(output_layer)
        goal = tf.mul(action, reward)

        # Optimizer.
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(probability,goal))
        policy_optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

        return observation, action, reward, probability, policy_optimizer

class Pong:
    def __init__(self): 
        self.input_dim = 80 * 80
        self.output_dim = 2
        self.gamma = 0.9
        self.env = gym.make('Pong-v0')
        self.tf_observation, self.tf_action, self.tf_reward, self.tf_probability, self.tf_policy_optimizer = policy_network()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables()) 

    def choose_action(self, state):
        probability = self.tf_probability.eval(feed_dict={self.tf_observation : [state]})[0]
        action = np.random.choice(self.output_dim, 1, p=probability)[0]
        action_row = np.zeros(self.output_dim)
        action_row[action] = 1
        return action+2, action_row

    def encode_state(self, state):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        state = state[35:195] # crop
        state = state[::2,::2,0] # downsample by factor of 2
        state[state == 144] = 0 # erase background (background type 1)
        state[state == 109] = 0 # erase background (background type 2)
        state[state != 0] = 1 # everything else (paddles, ball) just set to 1 
        new_state = state.astype(np.float).ravel()
        return new_state

    def training(self, state_seq, action_seq, reward):
        feed_dict = {self.tf_observation : state_seq, self.tf_action : action_seq, self.tf_reward : reward}
        self.sess.run(self.tf_policy_optimizer, feed_dict=feed_dict)

    def run(self, game):
        for i in range(game):
            state_seq = []
            action_seq = []
            prev_state = None
            observation = self.env.reset()
            discount_factor = 1
            score = [0, 0]
            for _ in range(2000):
                self.env.render()
                cur_state = self.encode_state(observation)
                diff = cur_state - prev_state if prev_state is not None else np.zeros(self.input_dim)
                prev_state = cur_state
                state_seq.append(diff)
                action, action_row = self.choose_action(diff)
                observation, reward, done, info = self.env.step(action)
                action_seq.append(action_row * discount_factor)
                discount_factor *= self.gamma
                if reward == 1:
                    # get a score
                    self.training(state_seq, action_seq, reward)
                    state_seq = []
                    action_seq = []
                    discount_factor = 1
                    prev_state = None
                    score[1] += 1
                elif reward == -1:
                    # > 50 means that at least one bounce occured 
                    if len(state_seq) > 50:
                        reward = 1
                        self.training(state_seq, action_seq, reward)
                    state_seq = []
                    action_seq = []
                    discount_factor = 1
                    prev_state = None
                    score[0] += 1
                if done:
                    print "Game %d is over" % (i+1)
                    print "The final score is %d to %d" % (score[0], score[1])
                    break

if __name__ == "__main__":
    pong = Pong();
    pong.run(int(sys.argv[1]))
