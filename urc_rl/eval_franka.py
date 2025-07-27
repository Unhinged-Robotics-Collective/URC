import argparse
import os
import pickle
import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# from go2_env import Go2Env
from franka_reach_task import FrankaReachTask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="franka_reach")
    parser.add_argument("-c", "--ckpt", type=int, default=100)
    parser.add_argument("-n", "--num_envs", type=int, default=2)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    # cfgs = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # print(cfgs)
    # env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    env_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    # reward_cfg["reward_scales"] = {}

    env = FrankaReachTask(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg={},
        reward_cfg={},
        command_cfg={},
        show_viewer=True
        )
    # env = FrankaReachTask(
    #     num_envs=1,
    #     env_cfg=env_cfg,
    #     obs_cfg=obs_cfg,
    #     reward_cfg=reward_cfg,
    #     command_cfg=command_cfg,
    #     show_viewer=True,
    # )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    num_steps = 400
    with torch.no_grad():
        while True:
            for step in range(num_steps):
                actions = policy(obs)
                obs, rews, dones, infos = env.step(actions)
            obs, _ = env.reset()


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
