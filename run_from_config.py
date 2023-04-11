import argparse

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
    torch_utils.random_seed(cfg["seed"])

    # Save Path
    exp_path = cfg.get_save_dir(
        parsed.base_save_dir,
        format_args=["env_name", "dataset"],
        arg_hash=True)
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
    parsed = parser.parse_args()

    run_experiment(parsed.config, parsed.id, parsed.base_save_dir)
