import argparse
from gailtf.baselines.common import set_global_seeds, tf_util as U
import gym, logging, sys
from gailtf.baselines import bench
import os.path as osp
from gailtf.baselines import logger
from gailtf.dataset.mujoco import Mujoco_Dset
from gailtf.dataset.sc2_dataset import SC2Dataset
import numpy as np
import ipdb
from pysc2.env import sc2_env
from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='sc2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=1)
    parser.add_argument('--expert_path', type=str, default='/home/xuan/pysc2-replay/map_race_data/Ascension to Aiur LE_Terran_Terran/')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate'], default='evaluate')
    # for evaluatation
    parser.add_argument('--stochastic_policy', type=bool, default=False)
    #  Mujoco Dataset Configuration
    parser.add_argument('--ret_threshold', help='the return threshold for the expert trajectories', type=int, default=0)
    parser.add_argument('--traj_limitation', type=int, default=np.inf)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=5)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['bc', 'trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.02)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=5e-3)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=2e9)
    # Behavior Cloning
    parser.add_argument('--pretrained', help='Use BC to pretrain', type=bool, default=False)
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()

def get_task_name(args):
    if args.algo == 'bc':
        task_name = 'behavior_cloning.'
        if args.traj_limitation != np.inf: task_name += "traj_limitation_%d."%args.traj_limitation
        task_name += args.env_id.split("-")[0]
    else:
        import time
        t = time.strftime("%c")
        t = t.replace(' ','_')
        t = t.replace(':','_')
        task_name = args.algo + "_ppo_modify_available_action_gail." + t
        if args.pretrained: task_name += "with_pretrained."
        if args.traj_limitation != np.inf: task_name += "traj_limitation_%d."%args.traj_limitation
        task_name += args.env_id.split("-")[0]
        if args.ret_threshold > 0: task_name += ".return_threshold_%d" % args.ret_threshold
        task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
                ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    return task_name

def main(args):
    from gailtf.baselines.ppo1 import mlp_policy
    U.make_session(num_cpu=args.num_cpu).__enter__()
    # set_global_seeds(args.seed)
    # env = gym.make(args.env_id)

    MAP_USED = "'Ascension to Aiur LE'"
    RACE_USED = "Terran"

    env = sc2_env.SC2Env(
        map_name= 'AscensiontoAiur',
        agent_race="T", #Terran
        bot_race="T",
        difficulty=1,
        step_mul=8,
        screen_size_px=(64,64), # will change to (64,64)
        minimap_size_px=(64,64),
        visualize=True) 

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            reuse=reuse, hid_size=600, num_hid_layers=2) # 600 > 524
    # env = bench.Monitor(env, logger.get_dir() and
    #     osp.join(logger.get_dir(), "monitor.json"))
    # env.seed(args.seed)
    # gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    # dataset = Mujoco_Dset(expert_path=args.expert_path, ret_threshold=args.ret_threshold, traj_limitation=args.traj_limitation)
    dataset = SC2Dataset(expert_path=args.expert_path)
    pretrained_weight = None
    # if (args.pretrained and args.task == 'train') or args.algo == 'bc':
    #     # Pretrain with behavior cloning
    #     from gailtf.algo import behavior_clone
    #     if args.algo == 'bc' and args.task == 'evaluate':
    #         behavior_clone.evaluate(env, policy_fn, args.load_model_path, stochastic_policy=args.stochastic_policy)
    #         sys.exit()
    #     pretrained_weight = behavior_clone.learn(env, policy_fn, dataset,
    #         max_iters=args.BC_max_iter, pretrained=args.pretrained, 
    #         ckpt_dir=args.checkpoint_dir, log_dir=args.log_dir, task_name=task_name)
    #     if args.algo == 'bc':
    #         sys.exit()

    from gailtf.network.adversary import TransitionClassifier
    # discriminator
    discriminator = TransitionClassifier(args.adversary_hidden_size, entcoeff=args.adversary_entcoeff)
    if args.algo == 'trpo':
        # Set up for MPI seed
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        # env.seed(workerseed)
        from gailtf.algo import trpo_mpi
        if args.task == 'train':
            trpo_mpi.learn(env, policy_fn, discriminator, dataset,
                pretrained=args.pretrained, pretrained_weight=pretrained_weight,
                g_step=args.g_step, d_step=args.d_step,
                timesteps_per_batch=32, 
                max_kl=args.max_kl, cg_iters=10, cg_damping=0.1,
                max_timesteps=args.num_timesteps, 
                entcoeff=args.policy_entcoeff, gamma=0.99, lam=0.95, 
                vf_iters=3, vf_stepsize=5e-4,
                ckpt_dir=args.checkpoint_dir, log_dir=args.log_dir,
                save_per_iter=args.save_per_iter, load_model_path=args.load_model_path,
                task_name=task_name)
        elif args.task == 'evaluate':
            # trpo_mpi.evaluate(env, policy_fn, args.checkpoint_dir, timesteps_per_batch=1024,
            #     number_trajs=10, stochastic_policy=args.stochastic_policy)

            trpo_mpi.evaluate(env, policy_fn, 
                'checkpoint/trpo_gail.sc2.g_step_10.d_step_1.policy_entcoeff_0.01.adversary_entcoeff_0.001-300', 
                timesteps_per_batch=1024,
                number_trajs=10, 
                stochastic_policy=False) #args.stochastic_policy

        else: raise NotImplementedError
    else: raise NotImplementedError

    env.close()

if __name__ == '__main__':
    args = argsparser()
    main(args)
