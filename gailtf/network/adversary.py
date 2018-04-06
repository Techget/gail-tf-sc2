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
    self.msize = 60 # change to 64 later
    self.ssize = 60 
    self.isize = 11
    self.available_action_size = 523
    self.observation_shape = (5*self.msize*self.msize + 9*self.ssize*self.ssize + self.isize + self.available_action_size,) # minimap, screen, info, available_actions
    self.actions_shape = (1,) # actions argument, one value, range in (0, 523)

    self.input_shape = tuple([o+a for o,a in zip(self.observation_shape, self.actions_shape)])
    # self.input_shape = tuple(list(self.observation_shape).extend(self.action_space))

    # self.num_actions = env.action_space.shape[0]
    self.hidden_size = hidden_size

    self.build_ph()

    
    # self.minimap = tf.placeholder(tf.float32, [None, 5, self.msize, self.msize], name='minimap')
    # self.screen = tf.placeholder(tf.float32, [None, 9, self.ssize, self.ssize], name='screen')
    # self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')
    # self.available_action = tf.placeholder(tf.float32, [None, self.available_action_size], name='available_action')

    # self.minimap = 


    # Build grpah
    generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
    expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
    # Build accuracy
    generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
    expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
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
    self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
    var_list = self.get_trainable_variables()
    self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph], 
                         self.losses + [U.flatgrad(self.total_loss, var_list)])

  def build_ph(self):
    self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
    self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
    self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
    self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

  def build_graph(self, obs_ph, acs_ph, reuse=False):
    with tf.variable_scope(self.scope):
      if reuse:
        tf.get_variable_scope().reuse_variables()

      # with tf.variable_scope("obfilter"):
      #     self.obs_rms = RunningMeanStd(shape=self.observation_shape)
      # obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std

      minimap = obs_ph[:, 0:5*self.msize*self.msize]
      screen = obs_ph[:, 5*self.msize*self.msize:9*self.ssize*self.ssize]
      info = obs_ph[:, (5*self.msize*self.msize+9*self.ssize*self.ssize):(5*self.msize*self.msize+9*self.ssize*self.ssize+self.isize)]
      available_action = obs_ph[:, (5*self.msize*self.msize+9*self.ssize*self.ssize+self.isize):(5*self.msize*self.msize+9*self.ssize*self.ssize+self.isize+self.available_action_size)]

      mconv1 = layers.conv2d(tf.reshape(minimap, [0,self.msize,self.msize,5]),
                   num_outputs=16,
                   kernel_size=5,
                   stride=1)
      mconv2 = layers.conv2d(mconv1,
                   num_outputs=32,
                   kernel_size=3,
                   stride=1)
      sconv1 = layers.conv2d(tf.reshape(screen, [0,self.ssize, self.ssize,9]),
                   num_outputs=16,
                   kernel_size=5,
                   stride=1)
      sconv2 = layers.conv2d(sconv1,
                   num_outputs=32,
                   kernel_size=3,
                   stride=1)
      info_fc = layers.fully_connected(layers.flatten(info),
                   num_outputs=256,
                   activation_fn=tf.tanh)

      aa_fc = layers.fully_connected(layers.flatten(available_action),
                   num_outputs=256,
                   activation_fn=tf.tanh)

      # feat_conv = tf.concat([mconv2, sconv2], axis=3)
      # spatial_action = layers.conv2d(feat_conv,
      #                                 num_outputs=1,
      #                                 kernel_size=1,
      #                                 stride=1,
      #                                 activation_fn=None,
      #                                 scope='spatial_action')
      # self.spatial_action = tf.nn.softmax(layers.flatten(spatial_action))

      # # Compute non spatial actions and value
      # feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc, aa_fc], axis=1)
      # feat_fc = layers.fully_connected(feat_fc,
      #                                  num_outputs=256,
      #                                  activation_fn=tf.nn.relu,
      #                                  scope='feat_fc')
      # self.non_spatial_action = layers.fully_connected(feat_fc,
      #                                             num_outputs=num_action,
      #                                             activation_fn=tf.nn.softmax,
      #                                             scope='non_spatial_action')
      # self.value = tf.reshape(layers.fully_connected(feat_fc,
      #                                           num_outputs=1,
      #                                           activation_fn=None,
      #                                           scope='value'), [-1])


      # _input = tf.concat([obs, acs_ph], axis=1) # concatenate the two input -> form a transition
      _input = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc, aa_fc, acs_ph], axis=1)
      p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
      p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
      logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
    return logits

  def get_trainable_variables(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

  def get_reward(self, obs, acs):
    sess = U.get_session()
    if len(obs.shape) == 1:
      obs = np.expand_dims(obs, 0)
    if len(acs.shape) == 1:
      acs = np.expand_dims(acs, 0)
    feed_dict = {self.generator_obs_ph:obs, self.generator_acs_ph:acs}
    reward = sess.run(self.reward_op, feed_dict)
    return reward

