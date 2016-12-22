# -*- coding: utf-8 -*-
import sys
import numpy as np
import tensorflow as tf
from termcolor import colored

def policy_network():
    input_dim  = 16
    output_dim = 4
    with tf.variable_scope("policy"):
        # Define input and output dimension.
        observation = tf.placeholder(tf.float32, shape=(None, input_dim))
        action = tf.placeholder(tf.float32, shape=(None, output_dim))
        reward = tf.placeholder(tf.float32)

        # Variables.
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim]))
        biases = tf.Variable(tf.zeros([output_dim]))

        # Training computation.
        output_layer = tf.matmul(observation, weights) + biases
        probability = tf.nn.softmax(output_layer)
        goal = tf.mul(action, reward)

        # Optimizer.
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(probability, goal))
        policy_optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

        return observation, action, reward, probability, policy_optimizer

class FrozenLake:
    def __init__(self): 
        self.input_dim = 16
        self.output_dim = 4
        self.env = Environment()
        self.tf_observation, self.tf_action, self.tf_reward, self.tf_probability, self.tf_policy_optimizer = policy_network()
        self.action_symbol = {0: u"↑", 1: u"↓", 2: u"←", 3: u"→"}
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables()) 

    def choose_action(self, state):
        probability = self.tf_probability.eval(feed_dict={self.tf_observation : [state]})[0]
        action = np.random.choice(self.output_dim, 1, p=probability)[0]
        action_row = np.zeros(self.output_dim)
        action_row[action] = 1
        return action, action_row

    def encode_state(self, state):
        state_row = np.zeros(self.input_dim)
        state_row[state] = 1
        return state_row

    def training(self, state_seq, action_seq, reward):
        feed_dict = {self.tf_observation : state_seq, self.tf_action : action_seq, self.tf_reward : reward}
        self.sess.run(self.tf_policy_optimizer, feed_dict=feed_dict)

    def run_episode(self, episode):
        for i_episode in range(episode):
            state = self.env.reset()
            state_seq = []
            action_seq = []
            for t in range(100):
                state = self.encode_state(state)
                action, action_row = self.choose_action(state)
                state_seq.append(state)
                state, reward, done = self.env.step(action)
                action_seq.append(action_row)
                if done:
                    if reward == 1:
                        self.training(state_seq, action_seq, reward)
                    break
        self.print_policy()

    def print_policy(self):
        self.env.reset()
        self.env.render()
        holes = [5, 7, 11, 12]
        goal = [15]
        for i in range(4):
            for j in range(4):
                offset = i * 4 + j
                if offset in holes:
                    print "H" ,
                    continue
                if offset in goal:
                    print "G" ,
                    continue
                state = np.zeros(self.input_dim)
                state[offset] = 1
                probability = self.tf_probability.eval(feed_dict={self.tf_observation : [state]})[0]
                idx = np.argmax(probability)
                print self.action_symbol[idx] ,
            print ""

class Environment:
    def __init__(self):
        self.position = 0
        self.width = 4
        self.height = 4
        self.grid = np.zeros(self.width * self.height)
        self.move_table = {0 : -self.width, 1 : self.width, 2 : -1, 3 : 1}
        self.hole = [5, 7, 11, 12]
        self.goal = [15]

    def reset(self):
        self.position = 0
        return self.position

    def step(self, action):
        done = 0
        reward = 0
        movement = self.move_table[action]
        if self.position + movement >= 0 and self.position + movement < self.width*self.height:
            self.position = self.position + movement
        if self.position in self.goal or self.position in self.hole:
            done = 1
        if self.position in self.goal:
            reward = 1
        return self.position, reward, done

    def render(self):
        for idx_i in range(self.height):
            for idx_j in range(self.width):
                if idx_i * self.width + idx_j == 0:
                    print "S" ,
                elif idx_i * self.width + idx_j in self.hole:
                    print "H" ,
                elif idx_i * self.width + idx_j in self.goal:
                    print "G" ,
                else:
                    print "F" ,
            print ""
        print ""

if __name__ == "__main__":
    fl = FrozenLake();
    fl.run_episode(int(sys.argv[1]))
