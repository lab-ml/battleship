import random
import math

from typing import List
from collections import namedtuple
from itertools import count

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from labml import tracker, monit, experiment, lab
from labml.helpers.training_loop import TrainingLoopConfigs
from labml.helpers.pytorch.device import DeviceConfigs
from labml.configs import option

from battleship.board import Board
from battleship.consts import EMPTY, BOMBED, SHIP, WON, SUNK_SHIP, BOARD_SIZE
from battleship.games import generate_games

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h: int = 10, w: int = 10, outputs: int = 100):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        convw = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(w)))
        convh = self.conv2d_size_out(self.conv2d_size_out(self.conv2d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    @staticmethod
    def conv2d_size_out(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 1


class Configs(TrainingLoopConfigs, DeviceConfigs):
    epochs: int = 50

    loop_step = 'loop_step'
    loop_count = 'loop_count'

    is_save_models = True
    batch_size: int = 128
    test_batch_size: int = 128

    use_cuda: bool = True
    cuda_device: int = 0
    seed: int = 5
    train_log_interval: int = 10

    is_log_parameters: bool = True

    device: any

    policy: nn.Module
    target: nn.Module

    learning_rate: float = 0.01
    optimizer: optim.Adam

    set_seed = 'set_seed'

    gamma: int = 0.999
    eps_start: int = 0.9
    eps_end: int = 0.05
    eps_decay: int = 200
    target_update: int = 10

    h: int = 10
    w: int = 10
    n_actions = 100

    memory: ReplayMemory
    episode_durations: List = []

    games = generate_games(epochs)

    @staticmethod
    # TODO change board methods according to this scoring
    def get_reward(res):
        if res == WON:
            return 10
        elif res == SUNK_SHIP:
            return 5
        elif res == SHIP:
            return 2
        elif res == EMPTY:
            return -2
        else:
            return -5

    def get_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * tracker.get_global_step() / self.eps_decay)

        tracker.add_global_step()

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    @staticmethod
    def unravel_index(index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()

        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def step(self, board: Board, action):
        num, let = self.unravel_index(action, [BOARD_SIZE, BOARD_SIZE])

        res = board.play(num, let)

        reward = torch.tensor([reward], device=self.device)

        return next_state, reward, done,

    def test(self):
        pass

    def run(self):
        for epoch, game in enumerate(self.games):
            board = Board(game)

            state = board.get_board()

            for t in count():
                action = self.get_action(state)
                next_state, reward, done = self.step(board, action.item())

                memory.push(state, action, next_state, reward)

                state = next_state

                self.train()

                if done:
                    self.episode_durations.append(t + 1)
                    plot_durations()
                    break

            if epoch % self.target_update == 0:
                self.target.load_state_dict(self.policy.state_dict())


@option(Configs.set_seed)
def set_seed(c: Configs):
    torch.manual_seed(c.seed)


@option(Configs.policy)
def policy(c: Configs):
    m: DQN = DQN(c.h, c.w, c.n_actions)
    m.to(c.device)
    return m


@option(Configs.target)
def target(c: Configs):
    m: DQN = DQN(c.h, c.w, c.n_actions)
    m.to(c.device)

    m.load_state_dict(c.policy.state_dict())

    return m


@option(Configs.optimizer)
def adam_optimizer(c: Configs):
    return optim.Adam(c.policy.parameters(), lr=c.learning_rate)


@option(Configs.memory)
def memory():
    return ReplayMemory(10000)


def main():
    conf = Configs()

    experiment.create(name='Battleship_DQN')
    experiment.calculate_configs(conf,
                                 {},
                                 ['set_seed', 'policy', 'target', 'run'])
    experiment.add_pytorch_models(dict(model=conf.policy))
    experiment.start()

    conf.run()

    experiment.save_checkpoint()


if __name__ == '__main__':
    main()
