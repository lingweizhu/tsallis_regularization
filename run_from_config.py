import os
import argparse
import toml

import core.environment.env_factory as environment
from core.utils import torch_utils, logger, run_funcs
from core.utils import config
from core import agent
from core.agent.in_sample import *
from core.agent.tsallis_inac import *
from core.construct import construct_agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--config", type=str)  # This is required
    parser.add_argument("--id", type=int)  # This is required
    parser.add_argument("--base_save_dir", type=str, default="./data/output/")
    parsed = parser.parse_args()
    cfg = config.Config(parsed.config, parsed.id)

    torch_utils.set_one_thread()
    torch_utils.random_seed(cfg["seed"])

    # project_root = os.path.abspath(os.path.dirname(__file__))
    exp_path = cfg.get_save_dir(parsed.base_save_dir, format_args=["env_name", "dataset"], arg_hash=True)

    torch_utils.ensure_dir(exp_path)
    env_fn = environment.EnvFactory.create_env_fn(cfg)
    offline_data = run_funcs.load_testset(
        cfg["env_name"], cfg["dataset"], cfg["seed"])
    # Setting up the logger
    lggr = logger.Logger(cfg, exp_path)
    # logger.log_config(cfg)
    cfg.log(lggr)

    # Initializing the agent and running the experiment
    agent_obj = construct_agent(cfg, env_fn, offline_data, lggr)
    run_funcs.run_steps(agent_obj, cfg.max_steps, cfg.log_interval, exp_path)
