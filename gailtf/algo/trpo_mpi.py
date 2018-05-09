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

import math

def extract_observation(time_step):
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

    unit_type = time_step.observation["screen"][6]
    unit_type_compressed = np.zeros(unit_type.shape, dtype=np.float)
    for y in range(len(unit_type)):
        for x in range(len(unit_type[y])):
            if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
                unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)

    hit_points = time_step.observation["screen"][8]
    hit_points_logged = np.zeros(hit_points.shape, dtype=np.float)
    for y in range(len(hit_points)):
        for x in range(len(hit_points[y])):
            if hit_points[y][x] > 0:
                hit_points_logged[y][x] = math.log(hit_points[y][x]) / 4

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
    output_ob.extend(list(state['available_actions']))

    output_ob = [output_ob]
    output_ob = np.array(output_ob)

    return state, output_ob



def traj_segment_generator(pi, env, discriminator, horizon, stochastic):
    # Initialize state variables
    t = 0
    # ac = env.action_space.sample()
    from gym import spaces
    ob_space = spaces.Box(low=-1000, high=10000, shape=(5*64*64 + 10*64*64 + 11 + 524,))
    ac_space = spaces.Discrete(524)
    ac = ac_space.sample()

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
    obs = np.array([ob for _ in range(horizon)])
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    param_sess = tf.Session(graph=original_graph, config=tf.ConfigProto(gpu_options=gpu_options))
    # sess.run(tf.global_variables_initializer())
    saved_model_path = os.path.expanduser('~')+'/pysc2-gail-research-project/supervised_learning_baseline/param_pred_model/action_params'
    # saver.restore(sess, saved_model_path+'action_params')

    with original_graph.as_default():
        saver = tf.train.import_meta_graph(saved_model_path+'.meta', clear_devices=True)
        saver.restore(param_sess,saved_model_path)

    # original_graph = tf.get_default_graph()

    # placeholder
    minimap_placeholder = original_graph.get_tensor_by_name("minimap_placeholder:0")
    screen_placeholder = original_graph.get_tensor_by_name("screen_placeholder:0")
    user_info_placeholder = original_graph.get_tensor_by_name("user_info_placeholder:0")
    action_placeholder = original_graph.get_tensor_by_name("action_placeholder:0")
    # ops

    # temp_2 = []
    # for n in original_graph.as_graph_def().node:
    #     if "minimap" in n.name:
    #         temp_2.append(n.name)

    # print(temp_2)
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
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "ep_true_rets": ep_true_rets}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = discriminator.get_reward(ob, ac)

        print('in traj_segment_generator, ac:', ac)

        # get action arguments with action_id
        function_type = sc_action.FUNCTIONS[ac].function_type.__name__
        one_hot_ac = np.zeros((1, 524)) # shape will be 1*254
        one_hot_ac[np.arange(1), [ac]] = 1
        ac_args = []


        # reshaped_minimap = 
        reshaped_minimap = np.reshape(np.array(state_dict['minimap']), (64,64,5))
        reshaped_screen = np.reshape(np.array(state_dict['screen']), (64,64,10))

        feed_dict = {minimap_placeholder: [reshaped_minimap], 
                screen_placeholder: [reshaped_screen], 
                action_placeholder: one_hot_ac, 
                user_info_placeholder: [state_dict['player']]}

        if function_type == 'move_camera':
            temp_arg1 = param_sess.run([minimap_output_pred], feed_dict)
            print('move_camera temp_arg1: ', temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'select_point':
            temp_arg1, temp_arg2 = param_sess.run([select_point_act_cls, screen_output_pred], feed_dict)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'select_rect':
            temp_arg1,temp_arg2, temp_arg3 = param_sess.run([select_add_pred_cls, screen_output_pred, screen2_output_pred],
                feed_dict)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
            ac_args.append(temp_arg3)
        elif function_type == 'select_unit':
            temp_arg1, temp_arg2 = param_sess.run([select_unit_act_cls, select_unit_id_output], feed_dict)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'control_group':
            temp_arg1, temp_arg2 = param_sess.run([control_group_act_cls, control_group_id_output], feed_dict)
            ac_args.append(temp_arg1)
            ac_args.append(temp_arg2)
        elif function_type == 'select_idle_worker':
            temp_arg1 = param_sess.run([select_worker_cls], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'select_army':
            temp_arg1 = param_sess.run([select_add_pred_cls], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'select_warp_gates':
            temp_arg1 = param_sess.run([select_add_pred_cls], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'unload':
            temp_arg1 = param_sess.run([unload_id_output], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'build_queue':
            temp_arg1 = param_sess.run([build_queue_id_output], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_quick':
            temp_arg1 = param_sess.run([queued_pred_cls], feed_dict)
            print('cmd_quick queued param:', temp_arg1)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_screen':
            temp_arg1, temp_arg2 = param_sess.run([queued_pred_cls, screen_output_pred], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'cmd_minimap':
            temp_arg1, temp_arg2 = param_sess.run([queued_pred_cls, minimap_output_pred], feed_dict)
            ac_args.append(temp_arg1)
        elif function_type == 'no_op' or function_type == 'select_larva' or function_type == 'autocast':
            # do nothing
            pass
        else:
            print("UNKNOWN FUNCTION TYPE: ", function_type)

        ac_with_param = sc_action.FunctionCall(ac, ac_args)
        timestep = env.step([ac_with_param])
        state_dict, ob = extract_observation(timestep[0])
        true_rew = timestep[0].reward
        new = timestep[0].last() # check is Done.
        # ob, true_rew, new, _ = 
        rews[i] = rew
        true_rews[i] = true_rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            timestep = env.reset()
            state_dict, ob = extract_observation(timestep[0])
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
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
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4, d_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        save_per_iter=100, ckpt_dir=None, log_dir=None, 
        load_model_path=None, task_name=None
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
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

    ob = U.get_placeholder_cached(name="ob")
    # ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None, ob_space[0]))
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    d_adam = MpiAdam(discriminator.get_trainable_variables())
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([U.sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

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
    d_adam.sync()
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, discriminator, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(discriminator.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # if provide pretrained weight
    if pretrained_weight is not None:
        U.load_state(pretrained_weight, var_list=pi.get_variables())
    # if provieded model path
    if load_model_path is not None:
        U.load_state(load_model_path)

    while True:        
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far)

        logger.log("********** Iteration %i ************"%iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p
        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)
            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            assign_old_eq_new() # set old parameter values to new parameter values
            with timed("computegrad"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg"):
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
                assert np.isfinite(stepdir).all()
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = get_flat()
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    set_from_flat(thbefore)
                if nworkers > 1 and iters_so_far % 20 == 0:
                    paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                    assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])
            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                    include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"): pi.ob_rms.update(mbob) # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)

        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = [] # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac), 
               include_final_partial_batch=False, batch_size=batch_size):
            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for discriminator
            if hasattr(discriminator, "obs_rms"): discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))
            *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
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
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
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

# Sample one trajectory (until trajectory end)
def traj_episode_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode

    # Initialize history arrays
    obs = []; rews = []; news = []; acs = []

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if t > 0 and (new or t % horizon == 0):
            # convert list into numpy array
            obs = np.array(obs)
            rews = np.array(rews)
            news = np.array(news)
            acs = np.array(acs)
            yield {"ob":obs, "rew":rews, "new":news, "ac":acs,
                    "ep_ret":cur_ep_ret, "ep_len":cur_ep_len}
            ob = env.reset()
            cur_ep_ret = 0; cur_ep_len = 0; t = 0

            # Initialize history arrays
            obs = []; rews = []; news = []; acs = []
        t += 1

def evaluate(env, policy_func, load_model_path, timesteps_per_batch, number_trajs=10, 
         stochastic_policy=False):
    
    from tqdm import tqdm
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=False)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    ep_gen = traj_episode_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
    U.load_state(load_model_path)

    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = ep_gen.__next__()
        ep_len, ep_ret = traj['ep_len'], traj['ep_ret']
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy: 
        print ('stochastic policy:')
    else:
        print ('deterministic policy:' )
    print ("Average length:", sum(len_list)/len(len_list))
    print ("Average return:", sum(ret_list)/len(ret_list))

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
