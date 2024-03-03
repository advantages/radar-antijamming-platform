import time
import sys
from tianshou.utils.net.common import Net
import os
import gymnasium as gym
import argparse
from tianshou.utils import TensorboardLogger
import numpy as np
import torch
from tianshou.env import DummyVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.trainer import offpolicy_trainer
import pprint
import sys
sys.path.append('../../jammer_environment')
sys.path.append('../../signal_simulate')
import load_param
import radarenv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='radar-game')      #  CartPole-v0
    parser.add_argument('--reward-threshold', type=float, default=2000)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=20)
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--testing-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_known_args()[0]
    return args



# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task', type=str, default='radar-game')      #  CartPole-v0
#     parser.add_argument('--reward-threshold', type=float, default=1000)
#     parser.add_argument('--seed', type=int, default=1626)
#     parser.add_argument('--eps-test', type=float, default=0.05)
#     parser.add_argument('--eps-train', type=float, default=0.1)
#     parser.add_argument('--buffer-size', type=int, default=320)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--gamma', type=float, default=0.9)
#     parser.add_argument('--n-step', type=int, default=3)
#     parser.add_argument('--target-update-freq', type=int, default=20)
#     parser.add_argument('--epoch', type=int, default=50)
#     parser.add_argument('--step-per-epoch', type=int, default=500)
#     parser.add_argument('--step-per-collect', type=int, default=100)
#     parser.add_argument('--update-per-step', type=float, default=0.1)
#     parser.add_argument('--batch-size', type=int, default=16)
#     parser.add_argument(
#         '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
#     )
#     parser.add_argument('--training-num', type=int, default=20)
#     parser.add_argument('--testing-num', type=int, default=20)
#     parser.add_argument('--test-num', type=int, default=1)
#     parser.add_argument('--logdir', type=str, default='log')
#     parser.add_argument('--render', type=float, default=0.)
#     parser.add_argument('--prioritized-replay', action="store_true", default=False)
#     parser.add_argument('--alpha', type=float, default=0.6)
#     parser.add_argument('--beta', type=float, default=0.4)
#     parser.add_argument(
#         '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
#     )
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
    # train_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.training_num)]
    # )
    #
    # test_envs = DummyVectorEnv(
    #     [lambda: gym.make(args.task) for _ in range(args.test_num)]
    # )

    train_envs = DummyVectorEnv(
        [lambda: radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len,select_version=0) for _ in range(args.training_num)]
    )

    test_envs = DummyVectorEnv(
        [lambda: radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len,select_version=0) for _ in range(args.test_num)]
    )

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

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    policy = DQNPolicy(
        net,
        optim,
        args.gamma,
        args.n_step,
        target_update_freq=args.target_update_freq,
    )

    buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    log_path = os.path.join(args.logdir, args.task, 'dqn')
    print(log_path)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)
    # train_collector.collect(10000)
    # sys.exit()
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    )
    # assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = radarenv.RadarGameEnv(episode=episode,num_sf=num_sf,num_sp=num_sp,jammer_type=jammer_type,history_len=his_len,select_version=1)
        policy.eval()
        policy.set_eps(args.eps_test)
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

