import argparse
import os
import pprint

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic
import time
import sys
sys.path.append('../../jammer_environment')
sys.path.append('../../signal_simulate')
import load_param
import radarenv

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='radar-game')
    parser.add_argument('--reward-threshold', type=float, default=1000)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128, 128])
    parser.add_argument('--training-num', type=int, default=256)
    parser.add_argument('--test-num', type=int, default=128)
    parser.add_argument('--logdir', type=str, default='log_new')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)
    args = parser.parse_known_args()[0]
    return args


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task', type=str, default='radar-game')
#     parser.add_argument('--reward-threshold', type=float, default=1000)
#     parser.add_argument('--seed', type=int, default=1626)
#     parser.add_argument('--buffer-size', type=int, default=20000)
#     parser.add_argument('--lr', type=float, default=3e-4)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--epoch', type=int, default=400)
#     parser.add_argument('--step-per-epoch', type=int, default=1000)
#     parser.add_argument('--step-per-collect', type=int, default=10)
#     parser.add_argument('--repeat-per-collect', type=int, default=10)
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
#     parser.add_argument('--training-num', type=int, default=256)
#     parser.add_argument('--test-num', type=int, default=128)
#     parser.add_argument('--logdir', type=str, default='log')
#     parser.add_argument('--render', type=float, default=0.)
#     parser.add_argument(
#         '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
#     )
#     # ppo special
#     parser.add_argument('--vf-coef', type=float, default=0.5)
#     parser.add_argument('--ent-coef', type=float, default=0.0)
#     parser.add_argument('--eps-clip', type=float, default=0.2)
#     parser.add_argument('--max-grad-norm', type=float, default=0.5)
#     parser.add_argument('--gae-lambda', type=float, default=0.95)
#     parser.add_argument('--rew-norm', type=int, default=0)
#     parser.add_argument('--norm-adv', type=int, default=0)
#     parser.add_argument('--recompute-adv', type=int, default=0)
#     parser.add_argument('--dual-clip', type=float, default=None)
#     parser.add_argument('--value-clip', type=int, default=0)
#     args = parser.parse_known_args()[0]
#     return args


def test_dqn(args=get_args()):
    episode, num_sf, num_sp, jammer_type, his_len = load_param.load_param("../../global_param/setting.json")

    env=radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len)
    # env=gym.make(args.task)



    args.state_shape = env.observation_space.shape
    args.action_shape = env.action_space.n
    # print(args.action_shape)
    # print(args.state_shape)
    # sys.exit()
    start_time = time.time()
    # train_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.training_num)]
    # )
    #
    # test_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.test_num)]
    # )

    train_envs = DummyVectorEnv(
        [lambda: radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len) for _ in range(args.training_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len,select_version=0) for _ in range(args.test_num)]
    )


    end_time = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        # dueling=(Q_param, V_param),
    ).to(args.device)

    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)

    actor_critic = ActorCritic(actor, critic)

    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)


    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    dist = torch.distributions.Categorical

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv
    )


    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)

    log_path = os.path.join(args.logdir, args.task, 'ppo_4_4_det1')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        step_per_collect=args.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )
    # assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # env = gym.make(args.task)
        # Let's watch its performance!
        env = radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len,select_version=1)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")




def test_pdqn(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    test_dqn(args)


if __name__ == '__main__':
    test_dqn(get_args())
















#
# end_time = time.time()
# print("程序执行时间为：", end_time - start_time, "秒")
#
# eng.quit()
