import sys

import torch
from Environment.SnakeEnvironment import SnakeEnvironment
device_name = 'cpu'
if device_name == 'cuda':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
elif device_name == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cpu')



num_episodes = sys.maxsize
max_time_steps = 500
batch_size = 16
render_mode = 'Human'
env = SnakeEnvironment(10, 10, 48, device=device, num_episodes=num_episodes, max_time_steps=max_time_steps, batch_size=batch_size, render_mode=render_mode)


