

import toml
import itertools
import hashlib
import os
import core.utils.torch_utils as torch_utils
from pathlib import Path

def load_toml_config(file_name):
    return toml.load(file_name)


def create_sweep_args(d):
    return [{k: v[idx] for idx, k in enumerate(d.keys())}
            for v in itertools.product(*d.values())]


class Config:
    _base_config = None
    _sweep_args = None
    args = None

    def __init__(self, file_name, id):
        self._base_config = load_toml_config(file_name)
        if self._base_config["config_version"] == 1:
            self._init_config_v1(id)
        else:
            raise "Config Version not Valid"

    def __getitem__(self, key):
        return self.args[key]

    def __getattr__(self, key):
        if str(key) in self.args.keys():
            return self.args[str(key)]
        else:
            return self.__dict__[key]

    def set_seed(self, seed):
        self.args["seed"] = seed

    def get_num_jobs(self):
        return len(self._sweep_args)

    def _init_config_v1(self, id):
        self._sweep_args = create_sweep_args(self._base_config["sweep"])
        self.args = {k: v for k, v in self._base_config.items()
                     if k not in ("sweep")}
        for k, v in self._sweep_args[id].items():
            self.args[k] = v

    def get_save_dir_and_save_config(
            self,
            base_dir,
            preformat_args,
            postformat_args,
            arg_hash,
            extra_hash_ignore=[],
            save_config=True):

        pre_args = [arg + "-" + str(self.args[arg]) for arg in preformat_args]
        arg_d = {k: v for k, v in self.args.items()
                 if k not in preformat_args and
                 k not in ["config_version"] and
                 k not in postformat_args and
                 k not in extra_hash_ignore}
        post_args = [arg + "-" + str(self.args[arg])
                     for arg in postformat_args]
        a_id = None
        if arg_hash:
            hasher = hashlib.sha1()
            hasher.update(str(arg_d).encode())
            a_id = hasher.hexdigest()
        else:
            srt_keys = list(arg_d.keys())
            srt_keys.sort()
            a_id = '_'.join([k + "-" + str(arg_d[k]) for k in srt_keys])

        if save_config:
            my_dir = Path(base_dir, Path(*pre_args), a_id)
            _cfg_file = Path(my_dir, '_'.join(post_args) + ".toml")
            torch_utils.ensure_dir(os.path.dirname(_cfg_file))
            with open(_cfg_file, "w") as f:
                toml.encoder.dump(self.args, f)

        return Path(base_dir, Path(*pre_args), a_id, Path(*post_args))

    def log(self, logger):
        cfg = self.args
        for param, value in cfg.items():
            logger.info('{}: {}'.format(param, value))
