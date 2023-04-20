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
from core.utils.Runner import Runner
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


def print_num_jobs(cfg_file):
    print(config.Config(parsed.config, 0).get_num_jobs())


def print_job_params(cfg_file, id, base_save_dir):
    lgr = logging.getLogger()
    lgr.setLevel(level=logging.INFO)
    cfg = config.Config(cfg_file, id)
    cfg.log(lgr)
    exp_path = cfg.get_save_dir_and_save_config(
        base_save_dir,
        preformat_args=["env_name", "dataset"],
        postformat_args=["run"],
        arg_hash=True,
        extra_hash_ignore=["seed", "run"], save_config=False)
    lgr.info('{}: {}'.format("SAVE DIR", exp_path))


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
        print_num_jobs(parsed.config)

    if parsed.get_job_params:
        print_job_params(parsed.config, parsed.id, parsed.base_save_dir)

    if not (parsed.get_num_jobs or parsed.get_job_params):
        runner = Runner(parsed.config,
                        parsed.id,
                        parsed.base_save_dir,
                        parsed.num_threads,
                        checkpoint=True)
        runner.run_experiment()
