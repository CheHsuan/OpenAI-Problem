# -*- coding: utf-8 -*-
import sys
import gym
import argparse
import numpy as np
import tensorflow as tf

class Pong:
    def __init__(self, path=None): 
        self.input_dim = 80 * 80
        self.hidden_dim = 200
        self.output_dim = 3
        self.gamma = 1
        self.env = gym.make('Pong-v0')
        self.tf_observation, self.tf_action, self.tf_reward, self.tf_df, self.tf_probability, self.tf_policy_optimizer = self.policy_network()
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        if path is None:
            self.sess.run(tf.initialize_all_variables())
        else:
            self.load(path)

    def policy_network(self):
        with tf.variable_scope("policy"):
            # Define input and output dimension.
            observation = tf.placeholder(tf.float32, shape=(None, self.input_dim))
            action = tf.placeholder(tf.float32, shape=(None, self.output_dim))
            reward = tf.placeholder(tf.float32)
            discount_factor = tf.placeholder(tf.float32, shape=(None, 1))

            # Variables.
            w1 = tf.Variable(tf.truncated_normal([self.input_dim, self.hidden_dim]))
            b1 = tf.Variable(tf.zeros([self.hidden_dim]))
            w2 = tf.Variable(tf.truncated_normal([self.hidden_dim, self.output_dim]))
            b2 = tf.Variable(tf.zeros([self.output_dim]))

            # Training computation.
            hidden_layer = tf.nn.relu(tf.matmul(observation, w1) + b1)
            output_layer = tf.matmul(hidden_layer, w2) + b2
            probability = tf.nn.softmax(output_layer)
            target = tf.mul(action, reward)

            # Optimizer.
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(probability, target)
            loss= tf.reduce_mean(tf.mul(cross_entropy, discount_factor))
            policy_optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

            return observation, action, reward, discount_factor, probability, policy_optimizer

    def save(self, path):
        self.saver.save(self.sess, path)

    def load(self, path):
        print "Load pretrained model"
        self.saver.restore(self.sess, path)

    def choose_action(self, state):
        probability = self.tf_probability.eval(feed_dict={self.tf_observation : [state]})[0]
        action = np.random.choice([1, 2, 3], 1, p=probability)[0]
        action_row = np.zeros(self.output_dim)
        action_row[action-1] = 1
        return action, action_row

    def encode_state(self, state):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        state = state[35:195] # crop
        state = state[::2,::2,0] # downsample by factor of 2
        state[state == 144] = 0 # erase background (background type 1)
        state[state == 109] = 0 # erase background (background type 2)
        state[state != 0] = 1 # everything else (paddles, ball) just set to 1
        new_state = state.astype(np.float).ravel()
        return new_state

    def training(self, state_seq, action_seq, df_seq, reward):
        feed_dict = {self.tf_observation : state_seq[:], self.tf_action : action_seq[:], self.tf_df : df_seq[:], self.tf_reward : reward}
        self.sess.run(self.tf_policy_optimizer, feed_dict=feed_dict)

    def run(self, game):
        for i in range(game):
            state_seq, action_seq, df_seq = [], [], []
            prev_state = None
            observation = self.env.reset()
            df = 1
            score = [0, 0]
            for _ in range(2000):
                self.env.render()
                cur_state = self.encode_state(observation)
                diff = cur_state - prev_state if prev_state is not None else np.zeros(self.input_dim)
                prev_state = cur_state
                action, action_row = self.choose_action(diff)
                observation, reward, done, info = self.env.step(action)
                action_seq.append(action_row)
                state_seq.append(diff)
                df_seq.append([df])
                df *= self.gamma
                if reward == 1:
                    # player gets a score
                    self.training(state_seq, action_seq, df_seq, reward)
                    state_seq, action_seq, df_seq = [], [], []
                    df = 1
                    prev_state = None
                    score[1] += 1
                elif reward == -1:
                    # > 50 means that at least one bounce occured
                    if len(state_seq) > 50:
                        self.training(state_seq[:50], action_seq[:50], df_seq[:50], 1)
                        self.training(state_seq[50:], action_seq[50:], df_seq[50:], -1)
                    else:
                        self.training(state_seq, action_seq, df_seq, reward)
                    state_seq, action_seq, df_seq = [], [], []
                    df = 1
                    prev_state = None
                    score[0] += 1
                if done:
                    print "Game %d is over. " % (i+1),
                    print "The final score is %d to %d" % (score[0], score[1])
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Checker')
    parser.add_argument("-n", "--episode", type=int, help="number of game", default=0)
    parser.add_argument("-o", "--output", type=str, help="trained model directory path")
    parser.add_argument("-i", "--input", type=str, help="pretrtained model directory path")
    args = parser.parse_args()
    pong = Pong(args.input);
    pong.run(args.episode)
    pong.save(args.output)
