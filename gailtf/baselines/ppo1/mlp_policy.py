from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import gailtf.baselines.common.tf_util as U
import tensorflow as tf
import gym
from gailtf.baselines.common.distributions import make_pdtype
import tensorflow.contrib.layers as layers
import numpy as np
from datetime import datetime
import random
import math

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        last_action = U.get_placeholder(shape=(None, 524), dtype=tf.float32, name="last_action_one_hot")
        self.msize = 64 # change to 64 later
        self.ssize = 64 
        self.isize = 11
        self.available_action_size = 524

        available_action = ob[:, (5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize+self.available_action_size)]
        # ob = ob[:,:-(self.available_action_size)]

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        # obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -20.0, 20.0)
        obz = (ob - self.ob_rms.mean) / self.ob_rms.std

        minimap = obz[:, 0:5*self.msize*self.msize]
        # minimap /= 2
        screen = obz[:, 5*self.msize*self.msize: 5*self.msize*self.msize+ 10*self.ssize*self.ssize]
        # screen /= 2
        info = obz[:, (5*self.msize*self.msize+10*self.ssize*self.ssize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize)]
        # info /= 2


        # get value prediction, crtic
        mconv1 = tf.layers.conv2d(
            inputs=tf.reshape(minimap, [-1,self.msize,self.msize,5]),
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="vffcmconv1")
        mpool1 = tf.layers.max_pooling2d(inputs=mconv1, pool_size=[2, 2], strides=2, name="vffcmpool1")
        mconv2 = tf.layers.conv2d(
            inputs=mpool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="vffcmconv2")
        mpool2 = tf.layers.max_pooling2d(inputs=mconv2, pool_size=[2, 2], strides=2, name="vffcmpool2")
        mpool2_flat = tf.reshape(mpool2, [-1, 16 * 16 * 64])

        sconv1 = tf.layers.conv2d(
            inputs=tf.reshape(screen, [-1,self.ssize, self.ssize,10]),
            filters=48,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="vffcsconv1")
        spool1 = tf.layers.max_pooling2d(inputs=sconv1, pool_size=[2, 2], strides=2, name="vffcspool1")
        sconv2 = tf.layers.conv2d(
            inputs=spool1,
            filters=80,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="vffcsconv2")
        spool2 = tf.layers.max_pooling2d(inputs=sconv2, pool_size=[2, 2], strides=2, name="vffcspool2")
        spool2_flat = tf.reshape(spool2, [-1, 16 * 16 * 80])

        info_fc = tf.layers.dense(inputs=layers.flatten(info),
                   units=8,
                   activation=tf.tanh,
                   name="vffcdense1")
        
        aa_fc = tf.layers.dense(inputs=layers.flatten(available_action),
                   units=32,
                   activation=tf.tanh,
                   name="vffcdense2")

        HIDDEN_SIZE = 128
        l1_action = tf.layers.dense(layers.flatten(last_action), 256, tf.nn.relu, name="vffclastactdense")
        input_to_rnn = tf.reshape(l1_action, [-1, 16, 16])
        action_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, 
            forget_bias=1.0, state_is_tuple=True, name="vffclstmcell")
        inputs_rnn = tf.unstack(input_to_rnn, num=16, axis=1, name="vffcunstack")
        rnn_outputs,rnn_state= tf.contrib.rnn.static_rnn(action_lstm_cell,
            inputs_rnn, dtype=tf.float32)
        l2_action = tf.layers.dense(rnn_state[-1], 
            128, tf.nn.tanh, name="vffclstmdense2")          # hidden layer
        last_acs_ph_lstm = tf.layers.dense(l2_action, 
            32, tf.nn.tanh, name="vffclstmdense3")

        vf_last_out = tf.concat([mpool2_flat, spool2_flat, info_fc, aa_fc, last_acs_ph_lstm], 
            axis=1, name="vffcconcat")
        vf_last_out = tf.nn.tanh(U.dense(vf_last_out, hid_size, 
            "vffcfinaldense", weight_init=U.normc_initializer(1.0)))

        # last_out = ob
        # for i in range(num_hid_layers):
        #     last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(vf_last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # get action id
        mconv1 = tf.layers.conv2d(
            inputs=tf.reshape(minimap, [-1,self.msize,self.msize,5]),
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="polmconv1")
        mpool1 = tf.layers.max_pooling2d(inputs=mconv1, pool_size=[2, 2], strides=2, name="polmpool1")
        mconv2 = tf.layers.conv2d(
            inputs=mpool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="polmconv2")
        mpool2 = tf.layers.max_pooling2d(inputs=mconv2, pool_size=[2, 2], strides=2, name="polmpool2")
        mpool2_flat = tf.reshape(mpool2, [-1, 16 * 16 * 64])

        sconv1 = tf.layers.conv2d(
            inputs=tf.reshape(screen, [-1,self.ssize, self.ssize,10]),
            filters=48,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="polsconv1")
        spool1 = tf.layers.max_pooling2d(inputs=sconv1, pool_size=[2, 2], strides=2, name="polspool1")
        sconv2 = tf.layers.conv2d(
            inputs=spool1,
            filters=80,
            kernel_size=[5, 5],
            padding="same",
            kernel_initializer=U.normc_initializer(0.01),
            activation=tf.nn.leaky_relu,
            name="polsconv2")
        spool2 = tf.layers.max_pooling2d(inputs=sconv2, pool_size=[2, 2], strides=2, name="polspool2")
        spool2_flat = tf.reshape(spool2, [-1, 16 * 16 * 80])

        info_fc = tf.layers.dense(inputs=layers.flatten(info),
                   units=8,
                   activation=tf.tanh,
                   name="poldense1")
        
        aa_fc = tf.layers.dense(inputs=layers.flatten(available_action),
                   units=32,
                   activation=tf.tanh,
                   name="poldense2")

        HIDDEN_SIZE = 128
        l1_action = tf.layers.dense(layers.flatten(last_action), 256, tf.nn.relu, name="pollastactdense")
        input_to_rnn = tf.reshape(l1_action, [-1, 16, 16])
        action_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=HIDDEN_SIZE, 
            forget_bias=1.0, state_is_tuple=True, name="pollstmcell")
        inputs_rnn = tf.unstack(input_to_rnn, num=16, axis=1, name="polunstack")
        rnn_outputs,rnn_state= tf.contrib.rnn.static_rnn(action_lstm_cell,
            inputs_rnn, dtype=tf.float32)
        l2_action = tf.layers.dense(rnn_state[-1], 
            128, tf.nn.tanh, name="pollstmdense2")          # hidden layer
        last_acs_ph_lstm = tf.layers.dense(l2_action, 
            32, tf.nn.tanh, name="pollstmdense3")
        
        last_out = tf.concat([mpool2_flat, spool2_flat, info_fc, aa_fc, last_acs_ph_lstm], 
            axis=1, name="polconcat")

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            last_out = U.dense(last_out, (pdtype.param_shape()[0])*2, "polfinaldense", U.normc_initializer(0.01))
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        #stochastic = tf.placeholder(dtype=tf.bool, shape=())
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(available_action), self.pd.mode(available_action))
        self.ac = ac
        self._act = U.function([stochastic, ob, last_action], [ac, self.vpred])

    def act(self, stochastic, ob, last_action, train_length=0):
        # input last_action is a number, convert to one-hot
        one_hot_last_action = []
        if type(last_action) is np.ndarray:
            depth = last_action.size
            one_hot_last_action = np.zeros((depth, 524))
            one_hot_last_action[np.arange(depth), last_action] = 1
        else:
            # one_hot_acs = tf.one_hot(indices, depth)
            one_hot_last_action = np.zeros(524)
            one_hot_last_action[last_action] = 1
            one_hot_last_action = [one_hot_last_action]

        # last action should be one host
        ac1, vpred1 =  self._act(stochastic, ob, one_hot_last_action)

        # epsilon greedy search
        random.seed(datetime.now())
        # increase 1500 can make the epsilon decay slower
        if stochastic and random.random() < (0.3 * math.exp(-train_length/1500)):
            # assume one available action
            available_act_one_hot = ob[0][-524:]
            available_act = []
            for i in range(0, len(available_act_one_hot)):
                if available_act_one_hot[i] == 1.0:
                    available_act.append(i)
            ac1 = random.choice(available_act)
            
        # print(ac1)
        return ac1, vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
