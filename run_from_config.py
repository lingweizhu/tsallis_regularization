import argparse
import random
import logging
import traceback
import pathlib
import signal
import torch
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


def timeout_handler(signum, frame):
    print('Job was forcibly quit with signal', signum)
    # Let Parallel know we were forced to quit without finishing.
    exit(22)  # 22 is non-reserved I think...


signal.signal(signal.SIGUSR1, timeout_handler)  # USR1 in SLURM will be used as pre=warning for cancel.
signal.signal(signal.SIGTERM, timeout_handler)  # Terminate
signal.signal(signal.SIGINT, timeout_handler)  # Interupt


def run_experiment(config_file, job_id, base_save_dir, num_threads):
    # Parse Config
    cfg = config.Config(config_file, job_id)

    # Setup
    torch.use_deterministic_algorithms(True)
    torch_utils.set_thread_count(num_threads)

    # set seed
    random.seed(cfg["run"])
    seed = random.randint(1, 1000000000)
    cfg.set_seed(seed)
    torch_utils.random_seed(cfg["seed"])

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

    # here we need to capture errors and still return an error if we fail (for gnu-parallel).
    try:
        run_funcs.run_steps(agent_obj,
                            cfg["max_steps"],
                            cfg["log_interval"],
                            exp_path)
    except ValueError as e:
        with open(pathlib.Path(exp_path, "except.out"), 'w') as f:
            f.write(str(e))
            f.write(traceback.format_exc())




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument("--config", type=str)  # This is required
    parser.add_argument("--id", type=int)  # This is required
    parser.add_argument("--base_save_dir", type=str, default="./data/output/")
    parser.add_argument("--get_num_jobs", action="store_true")
    parser.add_argument("--get_job_params", action="store_true")
    parser.add_argument("--num_threads", type=int, default=1)
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
        run_experiment(parsed.config, parsed.id, parsed.base_save_dir, parsed.num_threads)
