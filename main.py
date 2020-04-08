"""
Main function for training and evaluating MARL algorithms in NMARL envs
@author: Tianshu Chu
"""
# Tempotarily append to the python path to import ssd envs - will fix later
import sys
sys.path.append('..')
##########################################################################

import argparse
import configparser
import logging
import tensorflow as tf
import threading
from envs.cacc_env import CACCEnv
from envs.ssd_env import SSDEnv
#from envs.large_grid_env import LargeGridEnv
#from envs.real_net_env import RealNetEnv
from agents.models import IA2C, IA2C_FP, IA2C_CU, MA2C_NC, MA2C_IC3, MA2C_DIAL
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag,
                   plot_evaluation, plot_train)


def parse_args():
    default_base_dir = '/Users/tchu/Documents/rl_test/deeprl_dist/ia2c_grid_0.9'
    default_config_dir = './config/config_ia2c_grid.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(2000, 2500, 10)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0):
    scenario = config.get('scenario')
    if scenario.startswith('atsc'):
        if scenario.endswith('large_grid'):
            return LargeGridEnv(config, port=port)
        else:
            return RealNetEnv(config, port=port)
    elif scenario.startswith('cacc'):
        return CACCEnv(config)
    else:
        # NEW: SSD environments
        return SSDEnv(config)


def init_agent(env, config, total_step, seed):
    """
    paraneters
    ----------
    n_s_ls: list, one entry per agent; length of state vector in CACC/ATCS OR shape of state tensor for cleanup/harvest
    n_a_ls: list, one entry per agent; number of discrete actions available
    neighbor_mask: adjacency matrix (n_agent, n_agent); 1 if agents are adjacent, 0 otherwise.
    distance_mask: matrix (n_agent, n_agent); distance between each pair of agents in the graph
    coop_gamma: float/int; temporal discount factor
    total_step: integer; total number of training steps
    config: 'model config' parameters from file
    seed: seed for reproducability (initialisations etc)
    """
    if env.agent == 'ia2c':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc':
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_ic3':
        # this is actually CommNet
        return MA2C_IC3(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                        total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu':
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial':
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir) # creates data, log, model folders if they do not exist
    init_log(dirs['log']) # intialises logging, creates log file etc.
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data']) # copy the config file to the data/ directory
    config = configparser.ConfigParser()
    config.read(config_dir) # parse the .yml config file

    # init env
    env = init_env(config['ENV_CONFIG'])
    logging.info('Training on: %s action dim: %r, agent dim: %d' % (env.scenario, env.n_a_ls, env.n_agent))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    model = init_agent(env, config['MODEL_CONFIG'], total_step, seed) # returns agent class 

    print("before init_train()")
    # disable multi-threading for safe SUMO implementation
    summary_writer = tf.summary.FileWriter(dirs['log'], graph=model.sess.graph)
    trainer = Trainer(env, model, global_counter, summary_writer, output_path=dirs['data'])
    trainer.run()
    print("after init_train()")

    # save model
    final_step = global_counter.cur_step
    logging.info('Training: save final model at step %d ...' % final_step)
    model.save(dirs['model'], final_step)


def evaluate_fn(agent_dir, output_dir, seeds, port, demo):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file 
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
