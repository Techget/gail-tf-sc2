from gailtf.baselines.common import explained_variance, zipsame, dataset, Dataset, fmt_row
from gailtf.baselines import logger
import gailtf.baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time, os
from gailtf.baselines.common import colorize
from mpi4py import MPI
from collections import deque
from gailtf.baselines.common.mpi_adam import MpiAdam
from gailtf.baselines.common.cg import cg
from contextlib import contextmanager
from gailtf.common.statistics import stats
import ipdb
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import stopwatch

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.lib import features
from pysc2.lib import FUNCTIONS
# from pysc2.lib import static_data
from gym import spaces

# from baselines.common.mpi_adam import MpiAdam
from gailtf.baselines.common.mpi_moments import mpi_moments

import math


LAST_EXPERT_LOSS = 0.0
LAST_EXPERT_ACC = -1.0
LAST_ACTION = 0.01

UP_TO_STEP = 16 # have it learn to play in the very beginning
ITER_SOFAR_GLOBAL = 0

# NOTICE remove action did last time from available action
def extract_observation(time_step, last_action=None):
    state = {}

    state["minimap"] = [
        time_step.observation["minimap"][0] / 255,                  # height_map
        time_step.observation["minimap"][1] / 2,                    # visibility
        time_step.observation["minimap"][2],                        # creep
        time_step.observation["minimap"][3],                        # camera
        # (time_step.observation["minimap"][5] == 1).astype(int),     # own_units
        # (time_step.observation["minimap"][5] == 3).astype(int),     # neutral_units
        # (time_step.observation["minimap"][5] == 4).astype(int),     # enemy_units
        time_step.observation["minimap"][6]                         # selected
    ]

    # print(np.array(state['minimap'][0]).shape)

    # unit_type = time_step.observation["screen"][6]
    # unit_type_compressed = np.zeros(unit_type.shape, dtype=np.float)
    # for y in range(len(unit_type)):
    #     for x in range(len(unit_type[y])):
    #         if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
    #             unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)

    def unit_type_process(t):
        if t > 0 and t in static_data.UNIT_TYPES:
            return static_data.UNIT_TYPES.index(t) / len(static_data.UNIT_TYPES) # float(, NOTICE!!!
        else:
            return 0
    vfunc = np.vectorize(unit_type_process)
    unit_type_compressed = vfunc(time_step.observation["screen"][6])

    # hit_points = time_step.observation["screen"][8]
    # hit_points_logged = np.zeros(hit_points.shape, dtype=np.float)
    # for y in range(len(hit_points)):
    #     for x in range(len(hit_points[y])):
    #         if hit_points[y][x] > 0:
    #             hit_points_logged[y][x] = math.log(hit_points[y][x]) / 4
    def hit_points_process(t):
        if t > 0:
            return math.log(t) / 4
        else:
            return 0
    vfunc = np.vectorize(hit_points_process)
    hit_points_logged = vfunc(time_step.observation["screen"][8])

    state["screen"] = [
        time_step.observation["screen"][0] / 255,               # height_map
        time_step.observation["screen"][1] / 2,                 # visibility
        time_step.observation["screen"][2],                     # creep
        time_step.observation["screen"][3],                     # power
        # (time_step.observation["screen"][5] == 1).astype(int),  # own_units
        # (time_step.observation["screen"][5] == 3).astype(int),  # neutral_units
        # (time_step.observation["screen"][5] == 4).astype(int),  # enemy_units
        unit_type_compressed,                                   # unit_type
        time_step.observation["screen"][7],                     # selected
        hit_points_logged,                                      # hit_points
        time_step.observation["screen"][9] / 255,               # energy
        time_step.observation["screen"][10] / 255,              # shields
        time_step.observation["screen"][11]                     # unit_density
    ]

    # for i in range(0, len(state['screen'])):
    #     print(np.array(state['screen']).shape)

    state["player"] = time_step.observation["player"]
        
    state["available_actions"] = np.zeros(len(sc_action.FUNCTIONS))
    for i in time_step.observation["available_actions"]:
        state["available_actions"][i] = 1.0

    output_ob = []
    for x in state["minimap"]:
        output_ob.extend(list(x.flatten()))
    for x in state["screen"]:
        output_ob.extend(list(x.flatten()))
    output_ob.extend(list(state['player']))

    aa_list = list(state['available_actions'])
    if last_action != None and sum(aa_list) > 1:
        aa_list[last_action] = 0
    output_ob.extend(aa_list)
    # output_ob.extend(list(state['available_actions']))

    output_ob = [output_ob]
    output_ob = np.array(output_ob)

    return state, output_ob

def process_coordinates_param_nn_output(coordinate):
    coordinate = np.array(coordinate)
    coordinate = coordinate.flatten()
    
    coordinate[0] = int(coordinate[0])
    coordinate[1] = int(coordinate[1])
    # print(coordinate)

    coordinate = np.clip(coordinate, 0, 63)

    return coordinate

def flatten_param(param):
    param = np.array(param)
    param = param.flatten()
    return param

def traj_segment_generator(pi, env, discriminator, horizon, expert_dataset, stochastic):
    # Initialize state variables
    t = 0
    # ac = env.action_space.sample()
    from gym import spaces
    ob_space = spaces.Box(low=-1000, high=10000, shape=(5*64*64 + 10*64*64 + 11 + 524,))
    ac_space = spaces.Discrete(524)
    # ac = ac_space.sample()
    ac = 0

    new = True
    rew = 0.0
    true_rew = 0.0
    timestep = env.reset()

    state_dict, ob = extract_observation(timestep[0])

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob[0] for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()  

    # new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    # new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    # saver = tf.train.Saver()

    original_graph = tf.Graph()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    param_sess = tf.Session(graph=original_graph) #, config=tf.ConfigProto(gpu_options=gpu_options)
    # sess.run(tf.global_variables_initializer())
    saved_model_path = os.path.expanduser('~')+'/pysc2-gail-research-project/supervised_learning_baseline/param_pred_model/action_params'
    # saver.restore(sess, saved_model_path+'action_params')

    with original_graph.as_default():
        saver = tf.train.import_meta_graph(saved_model_path+'.meta', clear_devices=True)
        saver.restore(param_sess,saved_model_path)

    # placeholder
    minimap_placeholder = original_graph.get_tensor_by_name("minimap_placeholder:0")
    screen_placeholder = original_graph.get_tensor_by_name("screen_placeholder:0")
    user_info_placeholder = original_graph.get_tensor_by_name("user_info_placeholder:0")
    action_placeholder = original_graph.get_tensor_by_name("action_placeholder:0")
    # ops
    control_group_act_cls = original_graph.get_tensor_by_name("control_group_act_cls:0")
    screen_output_pred = original_graph.get_tensor_by_name("screen_param_prediction:0")
    minimap_output_pred = original_graph.get_tensor_by_name("minimap_param_prediction:0")
    screen2_output_pred = original_graph.get_tensor_by_name("screen2_param_prediction:0")
    queued_pred_cls = original_graph.get_tensor_by_name("queued_pred_cls:0")
    control_group_id_output = original_graph.get_tensor_by_name("control_group_id_output:0")
    select_point_act_cls = original_graph.get_tensor_by_name("select_point_act_cls:0")
    select_add_pred_cls = original_graph.get_tensor_by_name("select_add_pred_cls:0")
    select_unit_act_cls = original_graph.get_tensor_by_name("select_unit_act_cls:0")
    select_unit_id_output = original_graph.get_tensor_by_name("select_unit_id_output:0")
    select_worker_cls = original_graph.get_tensor_by_name("select_worker_cls:0")
    build_queue_id_output = original_graph.get_tensor_by_name("build_queue_id_output:0")
    unload_id_output = original_graph.get_tensor_by_name("unload_id_output:0")

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob, prevac, len(ep_lens))
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_true_rets": ep_true_rets}
            _, vpred = pi.act(stochastic, ob, prevac, len(ep_lens))
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            # ep_rets = []
            # ep_true_rets = []
            # ep_lens = []
        i = t % horizon
        obs[i] = ob[0] # change shape from (32, 1, 61975) to (32, 61975)
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # it is not accurate, since everytime, randomly use one, the loss is variant
        # ob_expert, ac_expert = expert_dataset.get_next_batch(1) # only need one, since ob and ac is also 1

        rew = discriminator.get_reward(ob, ac, prevac)
        # print("in traj_segment_generator rew: ", rew)
        global LAST_EXPERT_ACC,LAST_EXPERT_LOSS, ITER_SOFAR_GLOBAL
        if LAST_EXPERT_LOSS > 0:
            rew[0][0] += LAST_EXPERT_LOSS
            LAST_EXPERT_LOSS -= 0.01 # decay
        if LAST_EXPERT_ACC < 1.0 and LAST_EXPERT_ACC != -1.0:
            rew[0][0] += 1 - LAST_EXPERT_ACC
            LAST_EXPERT_ACC += 0.01 # decay
        if ITER_SOFAR_GLOBAL < 5:
            rew /= 5 # do not trust reward in the beginning

        # print("in traj_segment_generator rew: ", rew, LAST_EXPERT_LOSS, LAST_EXPERT_ACC)
        # rew += np.log(1 - LAST_EXPERT_ACC + 1e-8)

        # print('in traj_segment_generator, ac:', ac)

        # get action arguments with action_id
        function_type = sc_action.FUNCTIONS[ac].function_type.__name__
        one_hot_ac = np.zeros((1, 524)) # shape will be 1*254
        one_hot_ac[np.arange(1), [ac]] = 1
        ac_args = []

        reshaped_minimap = np.reshape(np.array(state_dict['minimap']), (64,64,5))
        reshaped_screen = np.reshape(np.array(state_dict['screen']), (64,64,10))

        feed_dict = {minimap_placeholder: [reshaped_minimap], 
                screen_placeholder: [reshaped_screen], 
                action_placeholder: one_hot_ac, 
                user_info_placeholder: [state_dict['player']]}

        if function_type == 'move_camera':
            temp_arg1 = param_sess.run([minimap_output_pred], feed_dict) # temp_arg1 is look like [[[x, y]]]
            # shape of minimap output is different from screen and screen2
            temp_arg1 = process_coordinates_param_nn_output(temp_arg1[0])
            ac_args.append(temp_arg1)
        elif function_type == 'select_point':
            temp_arg1, temp_arg2 = param_sess.run([select_point_act_cls, screen_output_pred], feed_dict)
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2)
            ac_args.append(temp_arg2)
        elif function_type == 'select_rect':
            temp_arg1,temp_arg2, temp_arg3 = param_sess.run([select_add_pred_cls, screen_output_pred, screen2_output_pred],
                feed_dict)
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2)
            ac_args.append(temp_arg2)
            temp_arg3 = process_coordinates_param_nn_output(temp_arg3)
            ac_args.append(temp_arg3)
        elif function_type == 'select_unit':
            temp_arg1, temp_arg2 = param_sess.run([select_unit_act_cls, select_unit_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg2 = flatten_param(temp_arg2)
            temp_arg2 = temp_arg2.astype(int)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'control_group':
            temp_arg1, temp_arg2 = param_sess.run([control_group_act_cls, control_group_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg2 = flatten_param(temp_arg2)
            temp_arg2 = temp_arg2.astype(int)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'select_idle_worker':
            temp_arg1 = param_sess.run([select_worker_cls], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'select_army':
            temp_arg1 = param_sess.run([select_add_pred_cls], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'select_warp_gates':
            temp_arg1 = param_sess.run([select_add_pred_cls], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'unload':
            temp_arg1 = param_sess.run([unload_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg1 = temp_arg1.astype(int)
            ac_args.append(temp_arg1)
        elif function_type == 'build_queue':
            temp_arg1 = param_sess.run([build_queue_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg1 = temp_arg1.astype(int)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_quick':
            temp_arg1 = param_sess.run([queued_pred_cls], feed_dict)
            # print('cmd_quick queued param:', temp_arg1)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_screen':
            temp_arg1, temp_arg2 = param_sess.run([queued_pred_cls, screen_output_pred], feed_dict)
            temp_arg1 = np.array(temp_arg1)
            temp_arg1 = temp_arg1.flatten()
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2)
            ac_args.append(temp_arg2)
        elif function_type == 'cmd_minimap':
            temp_arg1, temp_arg2 = param_sess.run([queued_pred_cls, minimap_output_pred], feed_dict)
            temp_arg1 = np.array(temp_arg1)
            temp_arg1 = temp_arg1.flatten()
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2[0])
            ac_args.append(temp_arg2)
        elif function_type == 'no_op' or function_type == 'select_larva' or function_type == 'autocast':
            # do nothing
            pass
        else:
            print("UNKNOWN FUNCTION TYPE: ", function_type)

        # print(ac_args)
        ac_with_param = sc_action.FunctionCall(ac, ac_args)
        timestep = env.step([ac_with_param])
        state_dict, ob = extract_observation(timestep[0], ac) # remove last action from available action
        true_rew = timestep[0].reward
        if true_rew == None:
            true_rew = 0
        # if true_rew == None:
        #     true_rew = 0
        new = timestep[0].last() # check is Done.
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        # print('######new, cur_ep_len, rew, true_rew:', new, cur_ep_len, rew, true_rew)
        global UP_TO_STEP
        if new or cur_ep_len >= UP_TO_STEP:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            if cur_ep_true_ret >= 1:
                with open("win.txt", "a+") as f:
                    f.write('win!!!!!!! {}'.format(cur_ep_true_ret))
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            timestep = env.reset()
            state_dict, ob = extract_observation(timestep[0])
            ac = 0 # in order to refresh last action
            UP_TO_STEP = np.minimum(UP_TO_STEP + 1, 1500)
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    print("seg['rew']: ", seg["rew"])
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, discriminator, expert_dataset,
        pretrained, pretrained_weight, *,
        g_step, d_step,
        timesteps_per_batch, # what to train on
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.001,
        cg_damping=1e-2,
        vf_stepsize=3e-4, d_stepsize=2e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None,
        save_per_iter=100, ckpt_dir=None, log_dir=None, 
        load_model_path=None, task_name=None,
        timesteps_per_actorbatch=32,
        clip_param=0.3, adam_epsilon=3e-5,
        optim_epochs=2, optim_stepsize=2e-4, optim_batchsize=32,schedule='linear'
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    print("##### nworkers: ",nworkers)
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)    
    # Setup losses and stuff
    # ----------------------------------------
    # ob_space = np.array([5*64*64 + 10*64*64 + 11 + 524]) # env.observation_space
    # ac_space = np.array([1]) #env.action_space
    from gym import spaces
    ob_space = spaces.Box(low=-1000, high=10000, shape=(5*64*64 + 10*64*64 + 11 + 524,))
    ac_space = spaces.Discrete(524)
    pi = policy_func("pi", ob_space, ac_space, reuse=(pretrained_weight!=None))
    oldpi = policy_func("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None, ob_space[0]))
    ac = pi.pdtype.sample_placeholder([None])
    # prevac = pi.pdtype.sample_placeholder([None])
    prevac_placeholder = U.get_placeholder_cached(name="last_action_one_hot")

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    # entbonus = entcoeff * meanent
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, prevac_placeholder, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    g_adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, prevac_placeholder, atarg, ret, lrmult], losses)

    # all_var_list = pi.get_trainable_variables()
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    d_adam = MpiAdam(discriminator.get_trainable_variables())
    # vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield
    
    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    writer = U.FileWriter(log_dir)
    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    g_adam.sync()
    d_adam.sync()
    # vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, discriminator, timesteps_per_batch, expert_dataset, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=100)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(discriminator.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # # if provide pretrained weight
    # if pretrained_weight is not None:
    #     U.load_state(pretrained_weight, var_list=pi.get_variables())
    # # if provieded model path
    # if load_model_path is not None:
    #     U.load_state(load_model_path)

    while True:        
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / (max_timesteps+1e7), 0.1)
            # cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        # Save model
        if iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far)

        logger.log("********** Iteration %i ************"%iters_so_far)

        # def fisher_vector_product(p):
        #     return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        meanlosses = []
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, prevac, atarg, tdlamret = seg["ob"], seg["ac"], seg["prevac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            # print("before standardize atarg value: ", atarg)
            if atarg.std() != 0:
                atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            # print("atarg value: ", atarg)

            # convert prevac to one hot
            one_hot_prevac = []
            if type(prevac) is np.ndarray:
              depth = prevac.size
              one_hot_prevac = np.zeros((depth, 524))
              one_hot_prevac[np.arange(depth), prevac] = 1
            else:
              one_hot_prevac = np.zeros(524)
              one_hot_prevac[prevac] = 1
              one_hot_prevac = [one_hot_prevac]
            prevac = one_hot_prevac

            d = Dataset(dict(ob=ob, ac=ac, prevac=prevac, atarg=atarg, vtarg=tdlamret), 
                shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]
            # print("optim_batchsize: ", optim_batchsize)

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            logger.log(fmt_row(13, loss_names))
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], 
                        batch["ac"], batch['prevac'], batch["atarg"], batch["vtarg"], cur_lrmult)
                    g_adam.update(g, optim_stepsize * cur_lrmult) # allmean(g)
                    x_newlosses = compute_losses(batch["ob"], batch["ac"], batch["prevac"],
                        batch["atarg"], batch["vtarg"], cur_lrmult)
                    meanlosses = [x_newlosses]
                    losses.append(x_newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

        # logger.log("Evaluating losses...")
        # losses = []
        # for batch in d.iterate_once(optim_batchsize):
        #     newlosses = compute_losses(batch["ob"], batch["ac"], batch["prevac"],
        #         batch["atarg"], batch["vtarg"], cur_lrmult)
        #     losses.append(newlosses)
        # meanlosses,_,_ = mpi_moments(losses, axis=0)

        # # logger.log("Evaluating losses...")
        # losses = []
        # for batch in d.iterate_once(optim_batchsize):
        #     newlosses = compute_losses(batch["ob"], batch["ac"], batch["prevac"],
        #         batch["atarg"], batch["vtarg"], cur_lrmult)
        #     losses.append(newlosses)
        # # # meanlosses,_,_ = mpi_moments(losses, axis=0) # it will be useful for multithreading
        meanlosses = np.mean(losses, axis=0)
        # logger.log(fmt_row(13, meanlosses))

        g_losses = meanlosses
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
       
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_name))
        global UP_TO_STEP
        ob_expert, ac_expert, prevac_expert = expert_dataset.get_next_batch(len(ob), UP_TO_STEP)
        batch_size = len(ob) // d_step
        d_losses = [] # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch, prevac_batch in dataset.iterbatches((ob, ac, prevac), 
               include_final_partial_batch=False, batch_size=batch_size):
            ob_expert, ac_expert, prevac_expert = expert_dataset.get_next_batch(len(ob_batch), UP_TO_STEP)
            # update running mean/std for discriminator
            if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            
            depth = len(ac_batch)
            one_hot_ac_batch = np.zeros((depth, 524))
            one_hot_ac_batch[np.arange(depth), ac_batch] = 1

            # depth = len(prevac_batch)
            # one_hot_prevac_batch = np.zeros((depth, 524))
            # one_hot_prevac_batch[np.arange(depth), prevac_batch] = 1

            depth = len(ac_expert)
            one_hot_ac_expert = np.zeros((depth, 524))
            one_hot_ac_expert[np.arange(depth), ac_expert] = 1

            depth = len(prevac_expert)
            one_hot_prevac_expert = np.zeros((depth, 524))
            one_hot_prevac_expert[np.arange(depth), prevac_expert] = 1

            *newlosses, g = discriminator.lossandgrad(ob_batch, one_hot_ac_batch, prevac_batch, ob_expert, one_hot_ac_expert, one_hot_prevac_expert)
            global LAST_EXPERT_ACC,LAST_EXPERT_LOSS
            LAST_EXPERT_ACC = newlosses[5]
            LAST_EXPERT_LOSS = newlosses[1]

            d_adam.update(g, d_stepsize) # allmean(g)
            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))


        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        # logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far = len(lens)
        timesteps_so_far = sum(lens) 
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)
            ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(rewbuffer),
                           np.mean(lenbuffer)], iters_so_far)

        global ITER_SOFAR_GLOBAL 
        ITER_SOFAR_GLOBAL = iters_so_far

        # log ac picked
        with open('ac.txt','a+') as f:
            print(ac, file=f)


def evaluate(env, policy_func, load_model_path, timesteps_per_batch, number_trajs=10, 
         stochastic_policy=False):
    # have it play with scripted bot for one full game
    ob_space = spaces.Box(low=-1000, high=10000, shape=(5*64*64 + 10*64*64 + 11 + 524,))
    ac_space = spaces.Discrete(524)
    pi = policy_func("pi", ob_space, ac_space, reuse=False)
    U.initialize()
    
    U.load_state(load_model_path)

    original_graph = tf.Graph()
    param_sess = tf.Session(graph=original_graph) 
    # saved_model_path = os.path.expanduser('~')+'/pysc2-gail-research-project/supervised_learning_baseline/param_pred_model/action_params'
    saved_model_path = 'param_pred_model/action_params'
    with original_graph.as_default():
        saver = tf.train.import_meta_graph(saved_model_path+'.meta', clear_devices=True)
        saver.restore(param_sess,saved_model_path)

    # placeholder
    minimap_placeholder = original_graph.get_tensor_by_name("minimap_placeholder:0")
    screen_placeholder = original_graph.get_tensor_by_name("screen_placeholder:0")
    user_info_placeholder = original_graph.get_tensor_by_name("user_info_placeholder:0")
    action_placeholder = original_graph.get_tensor_by_name("action_placeholder:0")
    # ops
    control_group_act_cls = original_graph.get_tensor_by_name("control_group_act_cls:0")
    screen_output_pred = original_graph.get_tensor_by_name("screen_param_prediction:0")
    minimap_output_pred = original_graph.get_tensor_by_name("minimap_param_prediction:0")
    screen2_output_pred = original_graph.get_tensor_by_name("screen2_param_prediction:0")
    queued_pred_cls = original_graph.get_tensor_by_name("queued_pred_cls:0")
    control_group_id_output = original_graph.get_tensor_by_name("control_group_id_output:0")
    select_point_act_cls = original_graph.get_tensor_by_name("select_point_act_cls:0")
    select_add_pred_cls = original_graph.get_tensor_by_name("select_add_pred_cls:0")
    select_unit_act_cls = original_graph.get_tensor_by_name("select_unit_act_cls:0")
    select_unit_id_output = original_graph.get_tensor_by_name("select_unit_id_output:0")
    select_worker_cls = original_graph.get_tensor_by_name("select_worker_cls:0")
    build_queue_id_output = original_graph.get_tensor_by_name("build_queue_id_output:0")
    unload_id_output = original_graph.get_tensor_by_name("unload_id_output:0")

    timesteps = env.reset()
    state_dict, ob = extract_observation(timesteps[0])
    is_done = False
    ac = 0

    while is_done == False:
        prevac = ac
        ac, vpred = pi.act(stochastic_policy, ob, prevac)
        function_type = sc_action.FUNCTIONS[ac].function_type.__name__
        one_hot_ac = np.zeros((1, 524)) # shape will be 1*254
        one_hot_ac[np.arange(1), [ac]] = 1
        ac_args = []

        reshaped_minimap = np.reshape(np.array(state_dict['minimap']), (64,64,5))
        reshaped_screen = np.reshape(np.array(state_dict['screen']), (64,64,10))

        feed_dict = {minimap_placeholder: [reshaped_minimap], 
                screen_placeholder: [reshaped_screen], 
                action_placeholder: one_hot_ac, 
                user_info_placeholder: [state_dict['player']]}

        if function_type == 'move_camera':
            temp_arg1 = param_sess.run([minimap_output_pred], feed_dict) # temp_arg1 is look like [[[x, y]]]
            # shape of minimap output is different from screen and screen2
            temp_arg1 = process_coordinates_param_nn_output(temp_arg1[0])
            ac_args.append(temp_arg1)
        elif function_type == 'select_point':
            temp_arg1, temp_arg2 = param_sess.run([select_point_act_cls, screen_output_pred], feed_dict)
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2)
            ac_args.append(temp_arg2)
        elif function_type == 'select_rect':
            temp_arg1,temp_arg2, temp_arg3 = param_sess.run([select_add_pred_cls, screen_output_pred, screen2_output_pred],
                feed_dict)
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2)
            ac_args.append(temp_arg2)
            temp_arg3 = process_coordinates_param_nn_output(temp_arg3)
            ac_args.append(temp_arg3)
        elif function_type == 'select_unit':
            temp_arg1, temp_arg2 = param_sess.run([select_unit_act_cls, select_unit_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg2 = flatten_param(temp_arg2)
            temp_arg2 = temp_arg2.astype(int)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'control_group':
            temp_arg1, temp_arg2 = param_sess.run([control_group_act_cls, control_group_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg2 = flatten_param(temp_arg2)
            temp_arg2 = temp_arg2.astype(int)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'select_idle_worker':
            temp_arg1 = param_sess.run([select_worker_cls], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'select_army':
            temp_arg1 = param_sess.run([select_add_pred_cls], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'select_warp_gates':
            temp_arg1 = param_sess.run([select_add_pred_cls], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'unload':
            temp_arg1 = param_sess.run([unload_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg1 = temp_arg1.astype(int)
            ac_args.append(temp_arg1)
        elif function_type == 'build_queue':
            temp_arg1 = param_sess.run([build_queue_id_output], feed_dict)
            temp_arg1 = flatten_param(temp_arg1)
            temp_arg1 = temp_arg1.astype(int)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_quick':
            temp_arg1 = param_sess.run([queued_pred_cls], feed_dict)
            # print('cmd_quick queued param:', temp_arg1)
            temp_arg1 = flatten_param(temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_screen':
            temp_arg1, temp_arg2 = param_sess.run([queued_pred_cls, screen_output_pred], feed_dict)
            temp_arg1 = np.array(temp_arg1)
            temp_arg1 = temp_arg1.flatten()
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2)
            ac_args.append(temp_arg2)
        elif function_type == 'cmd_minimap':
            temp_arg1, temp_arg2 = param_sess.run([queued_pred_cls, minimap_output_pred], feed_dict)
            temp_arg1 = np.array(temp_arg1)
            temp_arg1 = temp_arg1.flatten()
            ac_args.append(temp_arg1)
            temp_arg2 = process_coordinates_param_nn_output(temp_arg2[0])
            ac_args.append(temp_arg2)
        elif function_type == 'no_op' or function_type == 'select_larva' or function_type == 'autocast':
            # do nothing
            pass
        else:
            print("UNKNOWN FUNCTION TYPE: ", function_type)

        # print(ac_args)
        ac_with_param = sc_action.FunctionCall(ac, ac_args)
        print('take action with param: ', ac_with_param)
        timesteps = env.step([ac_with_param])
        print('env reward: ', timesteps[0].reward)
        state_dict, ob = extract_observation(timesteps[0], ac)
        is_done = timesteps[0].last()


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
