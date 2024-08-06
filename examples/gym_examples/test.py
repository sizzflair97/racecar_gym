import logging

from time import sleep
import gymnasium
import racecar_gym.envs.gym_api
from collections import OrderedDict

import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
from itertools import count

from tqdm.auto import tqdm

logging.basicConfig(filename="last.log", level=logging.INFO)
logger = logging.getLogger()

#https://medium.com/geekculture/policy-based-methods-for-a-continuous-action-space-7b5ecffac43a
class PGNet(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(PGNet, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1)


        self.fc1 = nn.Linear(in_features=133*64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions*2)

        self.relu = nn.ReLU()
        
        self.gaussian = torch.distributions.Normal(0, 1)

    def forward(self, x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        #batch, channels, features
        assert x.shape[-1] == 1098
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        mu, sigma = x[...,:2], nn.functional.softplus(x[...,2:])
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=-1)
        
        return action, log_prob
    

# Currently, three rendering modes are available: 'human', 'rgb_array_birds_eye' and 'rgb_array_follow'
# human: Render the scene in a window.
# rgb_array_birds_eye: Follow an agent in birds eye perspective.
# rgb_array_follow: Follow an agent in a 3rd person view.
env = gymnasium.make('SingleAgentAustria-v0', scenario='scenarios/custom.yml', render_mode=None)
policy_network = PGNet(1, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(policy_network.parameters())

for i_episode in (pbar:=tqdm(count(0))):
    done = False

    # Currently, there are two reset modes available: 'grid' and 'random'.
    # Grid: Place agents on predefined starting position.
    # Random: Random poses on the track.
    obs_tuple:tuple[dict] = env.reset(options=dict(mode='grid'))
    t = 0
    rewards:list = []
    log_prob_list:list = []
    
    for iteration in (pbar2:=tqdm(count(0), leave=False)):
        if done: break
        action = env.action_space.sample()
        obs = obs_tuple[0]
        net_input = np.concatenate([obs['pose'], obs['acceleration'], obs['velocity'], obs['lidar']])
        net_input = torch.tensor(net_input, dtype=torch.float32)[None,None,...]
        raw_out = policy_network(net_input)
        raw_action, log_prob = raw_out[0].detach().numpy(), raw_out[1]
        action = OrderedDict(motor=raw_action[:,0], steering=raw_action[:,1])
        obs, reward, done, truncated, states = env.step(action)
        rewards.append(reward)
        log_prob_list.append(log_prob)
        sleep(0.01)
        # if t % 30 == 0:
        #     image = env.render()
        t+=1
        pbar2.desc = f"{log_prob}"
        
    T = len(rewards)
    returns = np.empty(T, dtype=np.float32)
    future_return = 0.
    for t in range(T-1,-1,-1):
        future_return = rewards[t] + 1.0 * future_return
        returns[t] = future_return
    returns = torch.tensor(returns)
    log_probs = torch.stack(log_prob_list)
    loss = -log_probs * returns
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.desc = f"{loss}"
    logger.info(f"Return: {returns[0]}\tLoss: {loss}")
    
env.close()
