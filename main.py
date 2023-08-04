import torch
from Environment.SnakeEnvironment import SnakeEnvironment

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
num_episodes = 5000
max_time_steps = 500
batch_size = 32
env = SnakeEnvironment(8, 8, 40, device=device, num_episodes=num_episodes, max_time_steps=max_time_steps, batch_size=batch_size)


