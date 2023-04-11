

import toml
import itertools
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

    def _init_config_v1(self, id):
        self._sweep_args = create_sweep_args(self._base_config["sweep"])
        self.args = {k: v for k, v in self._base_config.items()
                     if k not in ("sweep")}
        for k, v in self._sweep_args[id].items():
            self.args[k] = v

    def get_save_dir(self, base_dir, format_args, arg_hash):
        p_args = [arg + "-" + str(self.args[arg]) for arg in format_args]
        arg_d = {k: v for k, v in self.args.items()
                 if k not in format_args and k not in ["config_version"]}
        a_id = None
        if arg_hash:
            a_id = str(hash(str(arg_d)))
        else:
            srt_keys = list(arg_d.keys())
            srt_keys.sort()
            a_id = '_'.join([k + "-" + str(arg_d[k]) for k in srt_keys])
        p_args.append(a_id)
        return Path(base_dir, *p_args)

    def log(self, logger):
        cfg = self.args
        for param, value in cfg.items():
            logger.info('{}: {}'.format(param, value))
