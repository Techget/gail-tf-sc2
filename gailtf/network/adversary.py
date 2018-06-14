from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
from gailtf.baselines.common import tf_util as U
from gailtf.common.tf_util import *
import numpy as np
import ipdb
import tensorflow.contrib.layers as layers

class TransitionClassifier(object):
  def __init__(self, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
    self.scope = scope
    # self.observation_shape = env.observation_space.shape
    # self.actions_shape = env.action_space.shape

    # print('~~~~~~~~~~', self.observation_shape, self.actions_space)
    self.msize = 64 # change to 64 later
    self.ssize = 64 
    self.isize = 11
    self.available_action_size = 524
    from gym import spaces
    self.ob_space = spaces.Box(low=-1000, high=10000, shape=(5*self.msize*self.msize + 10*self.ssize*self.ssize + self.isize + self.available_action_size,))
    self.ac_space = spaces.Discrete(self.available_action_size) 
    self.observation_shape = self.ob_space.shape
    self.actions_shape = self.ac_space.shape
    self.hidden_size = hidden_size

    self.build_ph() 

    # Build grpah
    generator_logits = self.build_graph(self.generator_obs_ph, 
      self.generator_acs_ph, self.generator_last_action_ph, reuse=False)
    expert_logits = self.build_graph(self.expert_obs_ph, 
      self.expert_acs_ph, self.expert_last_action_ph, reuse=True)
    # Build accuracy
    generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
    self.generator_acc = generator_acc
    expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
    self.expert_acc = expert_acc
    # Build regression loss
    # let x = logits, z = targets.
    # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
    generator_loss = tf.reduce_mean(generator_loss)
    expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
    expert_loss = tf.reduce_mean(expert_loss)
    # Build entropy loss
    logits = tf.concat([generator_logits, expert_logits], 0)
    entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
    entropy_loss = -entcoeff*entropy
    # Loss + Accuracy terms
    self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
    self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
    self.total_loss = generator_loss + expert_loss + entropy_loss
    # Build Reward for policy
    # make it larger, the network is large, it may vanish if reward is small
    # take generator_loss into consideration, since logits = 0.4 and logits equal to 0.1 are considered same otherwise
    self.reward_op = (-tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)+generator_loss)
    var_list = self.get_trainable_variables()
    self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.generator_last_action_ph, self.expert_obs_ph, self.expert_acs_ph, self.expert_last_action_ph], 
                         self.losses + [U.flatgrad(self.total_loss, var_list)])

  def build_ph(self):
    self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
    self.generator_acs_ph = tf.placeholder(tf.float32, (None, 524), name="actions_ph")
    self.generator_last_action_ph = tf.placeholder(tf.float32, (None, 524), name="last_actions_ph")
    self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
    self.expert_acs_ph = tf.placeholder(tf.float32, (None, 524), name="expert_actions_ph") #self.actions_shape
    self.expert_last_action_ph = tf.placeholder(tf.float32, (None, 524), name="expert_last_actions_ph")

  def build_graph(self, obs_ph, acs_ph, last_acs_ph, reuse=False):
    with tf.variable_scope(self.scope):
      if reuse:
        tf.get_variable_scope().reuse_variables()


      available_action = obs_ph[:, (5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize+self.available_action_size)]
      # obs_ph = obs_ph[:, :-524]

      with tf.variable_scope("obfilter"):
          self.obs_rms = RunningMeanStd(shape=self.observation_shape)
      obz = (obs_ph - self.obs_rms.mean) / self.obs_rms.std

      minimap = obz[:, 0:5*self.msize*self.msize]
      # minimap /= 2
      screen = obz[:, 5*self.msize*self.msize: 5*self.msize*self.msize+ 10*self.ssize*self.ssize]
      # screen /= 2
      info = obz[:, (5*self.msize*self.msize+10*self.ssize*self.ssize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize)]
      # info /= 2


      mconv1 = tf.layers.conv2d(
        inputs=tf.reshape(minimap, [-1,self.msize,self.msize,5]),
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.leaky_relu)
      mpool1 = tf.layers.max_pooling2d(inputs=mconv1, pool_size=[2, 2], strides=2)
      mconv2 = tf.layers.conv2d(
        inputs=mpool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.leaky_relu)
      mpool2 = tf.layers.max_pooling2d(inputs=mconv2, pool_size=[2, 2], strides=2)
      mpool2_flat = tf.reshape(mpool2, [-1, 16 * 16 * 64])

      sconv1 = tf.layers.conv2d(
        inputs=tf.reshape(screen, [-1,self.ssize, self.ssize,10]),
        filters=48,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.leaky_relu)
      spool1 = tf.layers.max_pooling2d(inputs=sconv1, pool_size=[2, 2], strides=2)
      sconv2 = tf.layers.conv2d(
        inputs=spool1,
        filters=80,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.leaky_relu)
      spool2 = tf.layers.max_pooling2d(inputs=sconv2, pool_size=[2, 2], strides=2)
      spool2_flat = tf.reshape(spool2, [-1, 16 * 16 * 80])

      info_fc = layers.fully_connected(layers.flatten(info),
                   num_outputs=8,
                   activation_fn=tf.tanh)

      aa_fc = layers.fully_connected(layers.flatten(available_action),
                   num_outputs=32,
                   activation_fn=tf.tanh)

      # _input = tf.concat([obs, acs_ph], axis=1) # concatenate the two input -> form a transition
      acs_ph_temp = tf.identity(acs_ph)
      acs_ph_temp = tf.expand_dims(acs_ph_temp, 1)
      # HIDDEN_SIZE = 128
      # l1_action = tf.layers.dense(layers.flatten(acs_ph_temp), 256, tf.nn.relu)
      # input_to_rnn = tf.reshape(l1_action, [-1, 16, 16])
      # action_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, forget_bias=1.0, state_is_tuple=True)
      # inputs_rnn = tf.unstack(input_to_rnn, num=16, axis=1)
      # rnn_outputs,rnn_state= tf.contrib.rnn.static_rnn(action_lstm_cell,inputs_rnn,dtype=tf.float32)
      # l2_action = tf.layers.dense(rnn_state[-1], 128, tf.nn.tanh)          # hidden layer
      # acs_ph_lstm = tf.layers.dense(l2_action, 32, tf.nn.tanh)
      acs_ph_dense_output = layers.fully_connected(layers.flatten(acs_ph_temp),
                   num_outputs=32,
                   activation_fn=tf.tanh)

      last_acs_ph_temp = tf.identity(last_acs_ph)
      last_acs_ph_temp = tf.expand_dims(last_acs_ph_temp, 1)
      last_acs_ph_dense_output = layers.fully_connected(layers.flatten(last_acs_ph_temp),
                   num_outputs=32,
                   activation_fn=tf.tanh)

      _input = tf.concat([mpool2_flat, spool2_flat, info_fc, aa_fc, acs_ph_dense_output, last_acs_ph_dense_output],
        axis=1)
      p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
      p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
      logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
    return logits

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

  def get_reward(self, obs, acs, last_acs):
    sess = U.get_session()
    # if len(obs.shape) == 1:
    #   obs = np.expand_dims(obs, 0)
    # if len(acs.shape) == 1:
    #   acs = np.expand_dims(acs, 0)

    one_hot_acs = []
    if type(acs) is np.ndarray:
      depth = acs.size
      one_hot_acs = np.zeros((depth, 524))
      one_hot_acs[np.arange(depth), acs] = 1
    else:
      # one_hot_acs = tf.one_hot(indices, depth)
      one_hot_acs = np.zeros(524)
      one_hot_acs[acs] = 1
      one_hot_acs = [one_hot_acs]

    one_hot_last_acs = []
    if type(last_acs) is np.ndarray:
      depth = last_acs.size
      one_hot_last_acs = np.zeros((depth, 524))
      one_hot_last_acs[np.arange(depth), last_acs] = 1
    else:
      one_hot_last_acs = np.zeros(524)
      one_hot_last_acs[last_acs] = 1
      one_hot_last_acs = [one_hot_last_acs]


    feed_dict = {self.generator_obs_ph:obs, 
      self.generator_acs_ph:one_hot_acs, self.generator_last_action_ph:one_hot_last_acs}
    # g_acc = sess.run(self.generator_acc, feed_dict)
    # reward = 0 
    # if g_acc > 0.99:
    #   reward = np.log(1-g_acc+1e-5) # give negative reward 
    # else:
    reward = sess.run(self.reward_op, feed_dict)

    if acs in [0,1,2,3,4,274]:
      reward /= 2

    # if reward < 0.01:
    #   # give negative reward
    #   reward = 10 * reward - 0.1

    # if np.allclose(reward, 0):
    #   reward = -1
    # if reward == 0:
    #   print('reward should not equal to 0!!!!!')
    return reward

