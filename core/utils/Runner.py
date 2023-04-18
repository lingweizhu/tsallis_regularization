

# def run_steps(agent, max_steps, log_interval, eval_pth):
#     t0 = time.time()
#     evaluations = []
#     agent.populate_returns(initialize=True)
#     while True:
#         if log_interval and not agent.total_steps % log_interval:
#             mean, median, min_, max_ = agent.log_file(
#                 elapsed_time=log_interval / (time.time() - t0), test=True)
#             evaluations.append(mean)
#             t0 = time.time()
#         if max_steps and agent.total_steps >= max_steps:
#             break
#         agent.step()
#     agent.save()
#     np.save(Path(eval_pth, "evaluations.npy"), np.array(evaluations))

# def run_experiment(config_file, job_id, base_save_dir, num_threads):
#     # Parse Config
#     cfg = config.Config(config_file, job_id)

#     # Setup
#     torch.use_deterministic_algorithms(True)
#     torch_utils.set_thread_count(num_threads)

#     # set seed
#     random.seed(cfg["run"])
#     seed = random.randint(1, 1000000000)
#     cfg.set_seed(seed)
#     torch_utils.random_seed(cfg["seed"])

#     # Save Path
#     exp_path = cfg.get_save_dir_and_save_config(
#         parsed.base_save_dir,
#         preformat_args=["env_name", "dataset"],
#         postformat_args=["run"],
#         arg_hash=True,
#         extra_hash_ignore=["seed", "run"])
#     torch_utils.ensure_dir(exp_path)

#     # DataSet and Environment loading
#     env_fn = environment.EnvFactory.create_env_fn(cfg)
#     offline_data = run_funcs.load_testset(
#         cfg["env_name"], cfg["dataset"], cfg["seed"])

#     # Setting up the logger
#     lggr = logger.Logger(cfg, exp_path)
#     cfg.log(lggr)

#     # Initializing the agent and running the experiment
#     agent_obj = construct_agent(
#         config=cfg,
#         exp_path=exp_path,
#         env_fn=env_fn,
#         offline_data=offline_data,
#         logger=lggr)


import os
import time
import random
import numpy as np
import torch
import traceback
import cloudpickle
import pickle
import copy


import core.environment.env_factory as environment

from pathlib import Path

from core.utils import config, logger, torch_utils
from core.construct import construct_agent
from core.utils.load_datasets import load_testset


class Runner(object):
    """Object that handles running experiments which followed

    the old run_steps protocol.
    """

    def __init__(self,
                 config_file,
                 job_id,
                 base_save_dir,
                 num_threads,
                 checkpoint=False):

        # Parse Config
        self.cfg = config.Config(config_file, job_id)
        self._initialize_comp_env(num_threads)

        self.save_path = self.cfg.get_save_dir_and_save_config(
            base_save_dir,
            preformat_args=["env_name", "dataset"],
            postformat_args=["run"],
            arg_hash=True,
            extra_hash_ignore=["seed", "run"])

        print(self.save_path)
        # check checkpointing
        chkpt_file = self._get_recent_checkpoint()
        print(chkpt_file)
        if checkpoint and chkpt_file is not None:
            self._initialize_from_checkpoint(chkpt_file)
            self._initialized_from_checkpoint = True
            return
        self._initialized_from_checkpoint = False
        # misc setup
        self.evaluations = []

        # Save Path

        torch_utils.ensure_dir(self.save_path)

        # DataSet and Environment loading
        env_fn = environment.EnvFactory.create_env_fn(self.cfg)
        offline_data = load_testset(
            self.cfg["env_name"], self.cfg["dataset"], self.cfg["seed"])

        # Setting up the logger
        self.lggr = logger.Logger(self.cfg, self.save_path)
        self.cfg.log(self.lggr)

        # Initializing the agent and running the experiment
        self.agent = construct_agent(
            config=self.cfg,
            exp_path=self.save_path,
            env_fn=env_fn,
            offline_data=offline_data,
            logger=self.lggr)

    def _initialize_comp_env(self, num_threads):
                # Setup
        torch.use_deterministic_algorithms(True)
        torch_utils.set_thread_count(num_threads)

        # set seed
        random.seed(self.cfg["run"])
        seed = random.randint(1, 1000000000)
        self.cfg.set_seed(seed)
        torch_utils.random_seed(self.cfg["seed"])

    def _chkpt_dir(self):
        return Path(self.save_path, "chkpt")

    def cleanup_checkpoints(self):
        chkpts = os.listdir(self._chkpt_dir())
        chkpts.sort()
        if chkpts is not None and len(chkpts) > 1:
            for c in chkpts[:-1]:
                os.remove(Path(self._chkpt_dir(), c))

    def _checkpoint_experiment(self, iteration):
        # Where to save self.save_path
        # iteration is the iteration we are saving on.

        # PICKLE EVERYTHING!
        cp_dict = {}
        cp_dict["version"] = "1"
        cp_dict["rngs"] = {}
        cp_dict["rngs"]["pytorch"] = torch.get_rng_state()
        cp_dict["rngs"]["python"] = random.getstate()
        cp_dict["rngs"]["numpy"] = np.random.get_state()
        cp_dict["evaluations"] = self.evaluations
        cp_dict["agent"] = self.agent  # This is the majority of work.

        torch_utils.ensure_dir(self._chkpt_dir())
        file_name = Path(self._chkpt_dir(),
                         "iteration_{}.pkl.tmp".format(iteration))

        with open(file_name, 'wb') as f:
            cloudpickle.dump(obj=cp_dict, file=f)

        os.rename(file_name,
                  Path(self._chkpt_dir(),
                       "iteration_{}.pkl".format(iteration)))
        # Once this is done, cleanup other checkpoints
        self.cleanup_checkpoints()

    def _initialize_from_checkpoint(self,
                                   chkpt_file):
        with open(chkpt_file, 'rb') as f:
            cp_dict = pickle.load(f)

        # initialize rngs
        torch.set_rng_state(cp_dict["rngs"]["pytorch"])
        random.setstate(cp_dict["rngs"]["python"])
        np.random.set_state(cp_dict["rngs"]["numpy"])

        # initialize other
        self.evaluations = cp_dict["evaluations"]
        self.agent = cp_dict["agent"]
        env_fn = environment.EnvFactory.create_env_fn(self.cfg)
        self.agent.env = env_fn()
        self.agent.eval_env = copy.deepcopy(env_fn)()
        # self.agent.eval_env = env_fn

    def _get_recent_checkpoint(self):
        ckpt_dir = self._chkpt_dir()
        try:
            ls = list(filter(lambda fn: fn[-4:] != ".tmp", os.listdir(ckpt_dir)))
        except FileNotFoundError:
            return None

        if len(ls) != 0:
            ls.sort()
            return Path(ckpt_dir, ls[-1])
        else:
            return None

    def log_and_eval(self, t0):
        mean, median, min_, max_ = self.agent.log_file(
            elapsed_time=self.cfg.log_interval / (time.time() - t0),
            test=True)
        self.evaluations.append(mean)

    def _run(self):
        t0 = time.time()

        agent = self.agent
        log_interval = self.cfg.log_interval
        max_steps = self.cfg.max_steps
        if not self._initialized_from_checkpoint:
            agent.populate_returns(initialize=True)

        while True:
            if log_interval and not agent.total_steps % log_interval:
                self._checkpoint_experiment(agent.total_steps)
                self.log_and_eval(t0)
                t0 = time.time()
            # if end

            if max_steps and agent.total_steps >= max_steps:
                break
            # if end
            agent.step()
        # while end

    def run_experiment(self):
        try:
            # run and save
            self._run()
            self.agent.save()
            np.save(Path(self.save_path, "evaluations.npy"),
                    np.array(self.evaluations))  # Final save.
        except ValueError as e:
            with open(Path(self.save_path, "except.out"), 'w') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            # if there is a value error, then we have nans somewhere.
            # save an evaluations that has nan.
            self.evaluations.append(float("nan"))
            np.save(Path(self.save_path, "evaluations.npy"),
                    np.array(self.evaluations))  # Final save.
