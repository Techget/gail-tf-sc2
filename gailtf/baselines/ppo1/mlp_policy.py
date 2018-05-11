from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import gailtf.baselines.common.tf_util as U
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gym
from gailtf.baselines.common.distributions import make_pdtype

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

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -10.0, 10.0)
        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]

        # last_out = obz
        # for i in range(num_hid_layers):
        #     last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        ### add conv net instead of using dense
        self.msize = 64 # change to 64 later
        self.ssize = 64 
        self.isize = 11
        self.available_action_size = 524
        minimap = obz[:, 0:5*self.msize*self.msize]
        screen = obz[:, 5*self.msize*self.msize: 5*self.msize*self.msize+ 10*self.ssize*self.ssize]
        info = obz[:, (5*self.msize*self.msize+10*self.ssize*self.ssize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize)]
        available_action = obz[:, (5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize+self.available_action_size)]

        conv1_minimap = tf.layers.conv2d(   
            inputs=tf.reshape(minimap, [-1,self.msize,self.msize,5]),
            filters=10,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.leaky_relu,
            name="polmconv1"
        )           # -> (64, 64, 10)
        pool1_minimap = tf.layers.max_pooling2d(
            conv1_minimap,
            pool_size=4,
            strides=4,
            name="polmpool1"
        )           # -> (16, 16, 10)
        conv2_minimap = tf.layers.conv2d(pool1_minimap, 10, 5, 1, 'same', activation=tf.nn.relu, name="polmconv2")    # -> (16, 16, 10)
        pool2_minimap = tf.layers.max_pooling2d(conv2_minimap, 2, 2, name="polmpool2")    # -> (8, 8, 10)
        flat_minimap = tf.reshape(pool2_minimap, [-1, 8*8*10])          # -> (8*8*10, )
        # dense_minimap = tf.layers.dense(inputs=flat_minimap, units=1024, activation=tf.nn.relu)
        # # dropout_mininmap = tf.layers.dropout(
        # #     inputs=dense_minimap, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        # minimap_output = tf.layers.dense(dense_minimap, 64)

        conv1_screen = tf.layers.conv2d(   
            inputs=tf.reshape(screen, [-1,self.ssize, self.ssize,10]), # (64,64,10)
            filters=20,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=tf.nn.leaky_relu,
            name="polsconv1"
        )           # -> (64, 64, 20)
        pool1_screen = tf.layers.max_pooling2d(
            conv1_screen,
            pool_size=4,
            strides=4,
            name="polspool1"
        )           # -> (16, 16, 20)
        conv2_screen = tf.layers.conv2d(pool1_screen, 20, 5, 1, 'same', activation=tf.nn.relu, name="polsconv2") # -> (16, 16, 20)
        pool2_screen = tf.layers.max_pooling2d(conv2_screen, 2, 2, name="polspool2")    # -> (8, 8, 20)
        flat_screen = tf.reshape(pool2_screen, [-1, 8*8*20])          # -> (8*8*20, )
        # dense_screen = tf.layers.dense(inputs=flat_screen, units=1024, activation=tf.nn.relu)
        # # dropout_screen = tf.layers.dropout(
        # #     inputs=dense_screen, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        # screen_output = tf.layers.dense(dense_screen, 64, tf.nn.relu)

        info_fc = tf.layers.dense(inputs=layers.flatten(info),
                   units=4,
                   activation=tf.tanh,
                   name="poldense1")
        
        aa_fc = tf.layers.dense(inputs=layers.flatten(available_action),
                   units=16,
                   activation=tf.tanh,
                   name="poldense2")

        last_out = tf.concat([flat_minimap, flat_screen, info_fc, aa_fc], axis=1, name="polconcat")
        # last_out = tf.layers.dense(inputs=last_out,units=600,name="poldense3")
        # last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc1", weight_init=U.normc_initializer(1.0)))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        #stochastic = tf.placeholder(dtype=tf.bool, shape=())
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        # print('~~~~',ob)

        available_act_one_hot = ob[0][-524:]
        # print(available_act_one_hot)
        available_act = []
        for i in range(0, len(available_act_one_hot)):
            if available_act_one_hot[i] == 1.0:
                available_act.append(i)
        # print('available_act int mlp_policy.py act function: ', available_act)
        # try to get valid action id,
        ac1, vpred1 =  self._act(stochastic, ob)
        while ac1[0] not in available_act:
            # print('try to loop to get action in available_act: ', ac1[0])
            ac1, vpred1 =  self._act(True, ob) # have to use stochastic

        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

