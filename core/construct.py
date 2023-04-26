from core.agent.tsallis_inac import TsallisInAC
from core.agent.in_sample import InSampleAC
import inspect
import sys


def construct_agent(config, **kwargs):
    an = config["agent_name"]
    local_funcs = dict(inspect.getmembers(sys.modules[__name__]))
    func_name = "construct_" + an
    if func_name in local_funcs.keys():
        return local_funcs[func_name](config, **kwargs)
    return locals()["construct_" + an](config, **kwargs)


def construct_TsallisInAC(config, exp_path, env_fn, offline_data, logger):
    return TsallisInAC(
        device=config["device"],
        discrete_control=config["discrete_control"],
        exp_path=exp_path,  # config["exp_path"],
        seed=config["seed"],
        timeout=config["timeout"],
        logger=logger,

        env_fn=env_fn,
        offline_data=offline_data,
        evaluation_criteria=config["evaluation_criteria"],

        state_dim=config["state_dim"],
        action_dim=config["action_dim"],

        hidden_units=config["hidden_units"],
        learning_rate=config["learning_rate"],

        batch_size=config["batch_size"],
        use_target_network=config["use_target_network"],
        target_network_update_freq=config["target_network_update_freq"],

        tau=config["tau"],
        polyak=config["polyak"],
        gamma=config["gamma"],
        q=config["q"],
    )


def construct_InSampleAC(config, exp_path, env_fn, offline_data, logger):
    return InSampleAC(
        device=config["device"],
        discrete_control=config["discrete_control"],
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        hidden_units=config["hidden_units"],
        learning_rate=config["learning_rate"],
        tau=config["tau"],
        polyak=config["polyak"],
        exp_path=exp_path,  # config["exp_path"],
        seed=config["seed"],
        env_fn=env_fn,
        timeout=config["timeout"],
        gamma=config["gamma"],
        offline_data=offline_data,
        batch_size=config["batch_size"],
        use_target_network=config["use_target_network"],
        target_network_update_freq=config["target_network_update_freq"],
        evaluation_criteria=config["evaluation_criteria"],
        logger=logger
    )    

