import os
import sys

import torch
from ModelInferenceEnvironment import SnakeEnvironment
device_name = 'cuda' #sys.argv[1]
if device_name == 'cuda':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
elif device_name == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cpu')



num_episodes = 10
max_time_steps = 500
batch_size = 16
render_mode = 'Human'
num_features = 512 #int(sys.argv[2])
with_hidden_layer = False #bool(sys.argv[3])
num_filters = 64 #int(sys.argv[4])
model_paths = os.listdir("../models")
# for path in model_paths:
#     env = SnakeEnvironment(10, 10, 50,
#                            device=device,
#                            num_episodes=num_episodes,
#                            max_time_steps=max_time_steps,
#                            batch_size=batch_size,
#                            render_mode=render_mode,
#                            model_path=os.path.join("../models", path)
#                            )


env = SnakeEnvironment(10, 10, 50,
                       device=device,
                       num_episodes=num_episodes,
                       max_time_steps=max_time_steps,
                       batch_size=batch_size,
                       render_mode=render_mode,
                       model_path=os.path.join("../models", "2270.pt")
                       )