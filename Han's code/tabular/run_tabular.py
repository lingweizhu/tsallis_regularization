import argparse
import copy

import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# from env import gridworld
from core.environment import gridworld
from core.tabular.dataset import *
from core.tabular.agents import *

np.set_printoptions(precision=3)
np.set_printoptions(linewidth=np.inf)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="run_tarbular")
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--discount', default=0.9, type=float)
  parser.add_argument('--policy', default='mixed', help='opt / mixed / random')
  parser.add_argument('--size', default=10000, type=int)
  parser.add_argument('--timeout', default=30, type=int)
  parser.add_argument('--iterations', default=10000, type=int)
  parser.add_argument('--tau', default=0.1, type=float)
  parser.add_argument('--initial', default=10.0, type=float)
  args = parser.parse_args()

  np.random.seed(args.seed)
  gw = gridworld.GridWorld(random_start=False)
  
  """
  Value iteration
  """
  opt_q = value_iteration(gw.P, gw.r, args.discount, args.iterations)
  # soft_opt_q = soft_value_iteration(gw.P, gw.r, args.tau, args.discount, args.iterations)

  """
  Data collection
  """
  if args.policy == 'opt':
    dataset = optimal_data_collection(gw, opt_q, args.size, args.timeout)
  elif args.policy == 'mixed':
    dataset = mixed_data_collection(gw, opt_q, args.size, args.timeout)
  elif args.policy == 'random':
    dataset = random_data_collection(gw, opt_q, args.size, args.timeout)
    # dataset = remove_selfloop(dataset)
  elif args.policy == 'missing_a':
    dataset = mixed_data_collection(gw, opt_q, args.size, args.timeout)
    # dataset = remove_action(dataset, [[0, 6], [0, 6]], 1)
    dataset = remove_action(dataset, [[0, 6], [0, 6]], 2)
    # dataset = remove_action(dataset, [[6, 6], [2, 3]], 0)
    # dataset = remove_action(dataset, [[6, 6], [2, 3]], 1)
    # dataset = remove_action(dataset, [[6, 6], [2, 3]], 3)
    # dataset = remove_selfloop(dataset)
  elif args.policy == 'missing_s':
    dataset = mixed_data_collection(gw, opt_q, args.size, args.timeout)
    dataset = remove_state(dataset, np.array(np.where(np.eye(6)==1)).T)
  else:
    raise NotImplementedError
  
  """
  Visualize dataset
  """
  # visualize_dataset(dataset, gw)
  
  """
  Dataset transition
  """
  beta, data_P, data_r, beta_next = behavior_policy(dataset, gw)
  
  """Chenjun's method"""
  in_sample_q, in_sample_pi = in_sample(copy.deepcopy(data_P), copy.deepcopy(data_r), copy.deepcopy(beta), args.tau, args.discount, args.iterations, init_constant=args.initial)
  """Baselines"""
  in_s_max_q, in_s_max_pi = in_sample_max(copy.deepcopy(data_P), copy.deepcopy(data_r), copy.deepcopy(beta), args.tau, args.discount, args.iterations, init_constant=args.initial)
  fqi_q, fqi_pi = fqi(copy.deepcopy(data_P), copy.deepcopy(data_r), copy.deepcopy(beta), args.discount, args.iterations, init_constant=args.initial)
  sarsa_q, sarsa_pi = sarsa(copy.deepcopy(data_P), copy.deepcopy(data_r), copy.deepcopy(beta), args.discount, args.iterations, init_constant=args.initial)
  # iql_q, iql_pi = iql(copy.deepcopy(data_P), copy.deepcopy(data_r), copy.deepcopy(beta), args.discount, args.iterations, init_constant=args.initial)
  # awac_q, awac_pi = awac(copy.deepcopy(data_P), copy.deepcopy(data_r), copy.deepcopy(beta), copy.deepcopy(beta_next), args.discount, args.iterations, init_constant=args.initial)
  
  """
  Visualize learned value
  """
  print("Optimal --------------")
  print(opt_q.max(axis=1).reshape((gw.num_rows, gw.num_cols)))
  # print("SoftMax --------------")
  # print(soft_opt_q.max(axis=1).reshape((env.num_rows, env.num_cols)))
  print("\n---------------------------- New ----------------------------")
  print(in_sample_q.max(axis=1).reshape((gw.num_rows, gw.num_cols)))
  print("\n---------------------------- Max ----------------------------")
  print(in_s_max_q.max(axis=1).reshape((gw.num_rows, gw.num_cols)))
  print("\n---------------------------- FQI ----------------------------")
  print(fqi_q.max(axis=1).reshape((gw.num_rows, gw.num_cols)))
  print("\n---------------------------- Sarsa ----------------------------")
  print(sarsa_q.max(axis=1).reshape((gw.num_rows, gw.num_cols)))
  # print("\n---------------------------- IQL ----------------------------")
  # print(iql_q.max(axis=1).reshape((gw.num_rows, gw.num_cols)))
  # print("\n---------------------------- AWAC ----------------------------")
  # print(awac_q.max(axis=1).reshape((13, 13)))
  
  # visualize_policy([in_sample_q, fqi_q, sarsa_q, iql_q], [in_sample_pi, fqi_pi, sarsa_pi, iql_pi], ['InSample', 'FQI', 'SARSA', 'IQL'], gw)
  visualize_policy([in_sample_q, in_s_max_q, fqi_q, sarsa_q], [in_sample_pi, in_s_max_pi, fqi_pi, sarsa_pi], ['InSample', 'InSampleMax', 'FQI', 'SARSA'], gw, args.policy)
