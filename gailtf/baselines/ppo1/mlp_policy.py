from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import gailtf.baselines.common.tf_util as U
import tensorflow as tf
import gym
from gailtf.baselines.common.distributions import make_pdtype
import tensorflow.contrib.layers as layers

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

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -20.0, 20.0)
        last_out = obz
        for i in range(1):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:,0]


        self.msize = 64 # change to 64 later
        self.ssize = 64 
        self.isize = 11
        self.available_action_size = 524
        minimap = obz[:, 0:5*self.msize*self.msize]
        screen = obz[:, 5*self.msize*self.msize: 5*self.msize*self.msize+ 10*self.ssize*self.ssize]
        info = obz[:, (5*self.msize*self.msize+10*self.ssize*self.ssize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize)]
        available_action = obz[:, (5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize):(5*self.msize*self.msize+10*self.ssize*self.ssize+self.isize+self.available_action_size)]

        # last_out = obz
        # for i in range(num_hid_layers):
        #     last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        # last_out_minimap = tf.nn.relu(U.conv2d(tf.reshape(minimap, [-1,self.msize,self.msize,5]),
        #     8, "l1-minimap", [8, 8], [4, 4], pad="VALID"))
        # last_out_minimap = U.flattenallbut0(last_out_minimap)
        # last_out_minimap = tf.nn.relu(U.dense(last_out_minimap, 128, 'lin-minimap', U.normc_initializer(1.0)))

        # last_out_screen = tf.nn.relu(U.conv2d(tf.reshape(screen, [-1,self.ssize, self.ssize,10]),
        #     16, "l1-screen", [8, 8], [4, 4], pad="VALID"))
        # last_out_screen = U.flattenallbut0(last_out_screen)
        # last_out_screen = tf.nn.relu(U.dense(last_out_screen, 350, 'lin-screen', U.normc_initializer(1.0)))


        mconv1 = tf.layers.conv2d(
            inputs=tf.reshape(minimap, [-1,self.msize,self.msize,5]),
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        mpool1 = tf.layers.max_pooling2d(inputs=mconv1, pool_size=[2, 2], strides=2)
        mconv2 = tf.layers.conv2d(
            inputs=mpool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        mpool2 = tf.layers.max_pooling2d(inputs=mconv2, pool_size=[2, 2], strides=2)
        mpool2_flat = tf.reshape(mpool2, [-1, 16 * 16 * 64])

        sconv1 = tf.layers.conv2d(
            inputs=tf.reshape(screen, [-1,self.ssize, self.ssize,10]),
            filters=48,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        spool1 = tf.layers.max_pooling2d(inputs=sconv1, pool_size=[2, 2], strides=2)
        sconv2 = tf.layers.conv2d(
            inputs=spool1,
            filters=80,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        spool2 = tf.layers.max_pooling2d(inputs=sconv2, pool_size=[2, 2], strides=2)
        spool2_flat = tf.reshape(spool2, [-1, 16 * 16 * 80])

        info_fc = tf.layers.dense(inputs=layers.flatten(info),
                   units=4,
                   activation=tf.tanh,
                   name="poldense1")
        
        aa_fc = tf.layers.dense(inputs=layers.flatten(available_action),
                   units=16,
                   activation=tf.tanh,
                   name="poldense2")

        last_out = tf.concat([mpool2_flat, spool2_flat, info_fc, aa_fc], axis=1, name="polconcat")

        # self.state_in = []
        # self.state_out = []

        # stochastic = tf.placeholder(dtype=tf.bool, shape=())
        # ac = self.pd.sample() # XXX
        # self._act = U.function([stochastic, ob], [ac, self.vpred])

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
