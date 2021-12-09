import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import model_urls
import gym
from collections import namedtuple, deque
from typing import Tuple
import numpy as np
from torch.utils.data.dataset import IterableDataset
from torchvision.models.resnet import ResNet, Bottleneck

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

def resnet():
    model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
    resnet50 = models.resnet50(pretrained=False)

    # resnet101 = models.resnet101(pretrained=True)
    # resnet152 = models.resnet152(pretrained=True)
    return resnet50

class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        num_classes = 1000
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))

class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)

def inception():
    inception = models.inception_v3(pretrained=True)

    return inception

def vgg():

    vgg11 = models.vgg11(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    return  (vgg11, vgg16)



def memoryCal():
    NotImplemented


# class ReplayBuffer:
#
#     def __init__(self, capacity: int) -> None:
#         self.buffer = deque(maxlen=capacity)
#
#     def __len__(self) -> None:
#         return len(self.buffer)
#
#     def append(self, transition: Transition) -> None:
#         self.buffer.append(transition)
#
#     def sample(self, batch_size: int) -> Tuple:
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         states, actions, rewards, dones, next_states = zip(*(self.buffer[idx] for idx in indices))
#
#         return (
#             np.array(states),
#             np.array(actions),
#             np.array(rewards, dtype=np.float32),
#             np.array(dones, dtype=np.bool),
#             np.array(next_states),)

# class DQN(nn.Module):
#
#     def __init__(self):
#         super(DQN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.head = nn.Linear(448, 2)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         return self.head(x.view(x.size(0), -1))
#     def save_checkpoint(self):
#         print('... saving checkpoint ...')
#         torch.save(self.state_dict(), self.checkpoint_file)
#
#     def load_checkpoint(self):
#         print('... loading checkpoint ...')
#         self.load_state_dict(torch.load(self.checkpoint_file))

# class RLDataset(IterableDataset):
#
#     def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
#         self.buffer = buffer
#         self.sample_size = sample_size
#
#     def __iter__(self) -> Tuple:
#         states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
#         for i in range(len(dones)):
#             yield states[i], actions[i], rewards[i], dones[i], new_states[i]

# class Agent:
#
#     def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
#
#         self.env = env
#         self.replay_buffer = replay_buffer
#         self.reset()
#         self.state = self.env.reset()
#
#     def reset(self) -> None:
#         self.state = self.env.reset()
#
#     def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
#         if np.random.random() < epsilon:
#             action = self.env.action_space.sample()
#         else:
#             state = torch.tensor([self.state])
#
#             if device not in ["cpu"]:
#                 state = state.cuda(device)
#
#             q_values = net(state)
#             _, action = torch.max(q_values, dim=1)
#             action = int(action.item())
#
#         return action
#
#     @torch.no_grad()
#     def play_step(
#         self,
#         net: nn.Module,
#         epsilon: float = 0.0,
#         device: str = "cpu",
#     ) -> Tuple[float, bool]:
#
#         action = self.get_action(net, epsilon, device)
#
#         # do step in the environment
#         new_state, reward, done, _ = self.env.step(action)
#
#         exp = Transition(self.state, action, reward, done, new_state)
#
#         self.replay_buffer.append(exp)
#
#         self.state = new_state
#         if done:
#             self.reset()
#         return reward, done

# def taxi():
#     NotImplemented
#     #https://github.com/eyalbd2/Deep_RL_Course
#     #https://data-newbie.tistory.com/547
# def frozenlake():
#     NotImplemented
#     #https://www.kaggle.com/wuhao1542/pytorch-rl-0-frozenlake-q-network-learning
#
# def cartpole():
#     NotImplemented
#     #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#
# def mountaincar():
#     NotImplemented
#     #https://github.com/huckiyang/DRL-torch-CoRL