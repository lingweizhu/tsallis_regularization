import gym
import d4rl


envs = ["halfcheetah-expert-v2",
        "halfcheetah-medium-expert-v2",
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        "walker2d-expert-v2",
        "walker2d-medium-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
        "hopper-expert-v2",
        "hopper-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "ant-expert-v2",
        "ant-medium-expert-v2",
        "ant-medium-v2",
        "ant-medium-replay-v2"]

for envname in envs:
    env = gym.make(envname)
    env.get_dataset()
