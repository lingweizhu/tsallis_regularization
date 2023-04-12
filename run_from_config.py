import argparse
import random
import logging
import core.environment.env_factory as environment
from core.utils import torch_utils, logger, run_funcs
from core.utils import config
from core.construct import construct_agent

# some compute canada concerns:
# - Saving data
# - Checkpointing
# - easy re-running.
# - Thunking
#   - Capture error messages
# - maybe mem-map of experiment config?
# - Check to see if experiment is already run?


def run_experiment(config_file, job_id, base_save_dir):
    # Parse Config
    cfg = config.Config(config_file, job_id)

    # Setup
    torch_utils.set_one_thread()
    # set seed
    random.seed(cfg["run"])
    seed = random.randint(1, 1000000000)
    cfg.set_seed(seed)
    # cfg["seed"] = seed
    torch_utils.random_seed(cfg["run"])

    # Save Path
    exp_path = cfg.get_save_dir_and_save_config(
        parsed.base_save_dir,
        preformat_args=["env_name", "dataset"],
        postformat_args=["run"],
        arg_hash=True,
        extra_hash_ignore=["seed", "run"])
    torch_utils.ensure_dir(exp_path)

    # DataSet and Environment loading
    env_fn = environment.EnvFactory.create_env_fn(cfg)
    offline_data = run_funcs.load_testset(
        cfg["env_name"], cfg["dataset"], cfg["seed"])

    # Setting up the logger
    lggr = logger.Logger(cfg, exp_path)
    cfg.log(lggr)

    # Initializing the agent and running the experiment
    agent_obj = construct_agent(
        config=cfg,
        exp_path=exp_path,
        env_fn=env_fn,
        offline_data=offline_data,
        logger=lggr)

    run_funcs.run_steps(agent_obj,
                        cfg["max_steps"],
                        cfg["log_interval"],
                        exp_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--config", type=str)  # This is required
    parser.add_argument("--id", type=int)  # This is required
    parser.add_argument("--base_save_dir", type=str, default="./data/output/")
    parser.add_argument("--get_num_jobs", action="store_true")
    parser.add_argument("--get_job_params", action="store_true")
    parsed = parser.parse_args()

    if parsed.get_num_jobs:
        print(config.Config(parsed.config, parsed.id).get_num_jobs())

    if parsed.get_job_params:
        lgr = logging.getLogger()
        lgr.setLevel(level=logging.INFO)
        cfg = config.Config(parsed.config, parsed.id)
        cfg.log(lgr)
        # config.Config(parsed.config, parsed.id).log(lgr)
        exp_path = cfg.get_save_dir_and_save_config(
            parsed.base_save_dir,
            preformat_args=["env_name", "dataset"],
            postformat_args=["run"],
            arg_hash=True,
            extra_hash_ignore=["seed", "run"], save_config=False)
        lgr.info('{}: {}'.format("SAVE DIR", exp_path))
        
    if not (parsed.get_num_jobs or parsed.get_job_params):
        run_experiment(parsed.config, parsed.id, parsed.base_save_dir)
    
