import sys

import torch
from Environment.SnakeEnvironment import SnakeEnvironment
device_name = 'cuda' #sys.argv[1]
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
num_features = 512 #int(sys.argv[2])
with_hidden_layer = False #bool(sys.argv[3])
num_filters = 64 #int(sys.argv[4])
model_directory = "C:/Users/Allan/Desktop/Models/SnakeModels"
env = SnakeEnvironment(10, 10, 50,
                       device=device,
                       num_episodes=num_episodes,
                       max_time_steps=max_time_steps,
                       batch_size=batch_size,
                       num_features=num_features,
                       render_mode=render_mode,
                       with_hidden_layer=with_hidden_layer,
                       num_filters=num_filters,
                       model_directory=model_directory
                       )


