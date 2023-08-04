import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, device, action_size, num_filters):
        super(Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=16, kernel_size=(3,3), stride=(1,1), device=device) #Batch norm the conv layers
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), device=device)
        #self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), device=device)


        self.features_layer = nn.Linear(in_features=3200, out_features=128, device=device) # Batch norm this layer
        self.output_layer = nn.Linear(in_features=128, out_features=action_size, device=device)

        self.conv_act1 = nn.ReLU()
        self.bn2d1 = nn.BatchNorm2d(num_features=16)
        self.conv_act2 = nn.ReLU()
        self.bn2d2 = nn.BatchNorm2d(num_features=32)
        # self.conv_act3 = nn.ReLU()
        # self.bn2d3 = nn.BatchNorm2d(num_features=32)

        self.features_act = nn.ReLU()
        self.features_bn = nn.BatchNorm1d(num_features=128)

        self.image_padding = (1,1,1,1)
        pass

    def forward(self, state):
        state = F.pad(state, self.image_padding)
        x = self.conv_act1(self.conv1(state))
        x = self.bn2d1(x)

        x = F.pad(x, self.image_padding)
        x = self.conv_act2(self.conv2(x))
        x = self.bn2d2(x)

        # x = F.pad(x, self.image_padding)
        # x = self.conv_act3(self.conv3(x))
        # x = self.bn2d3(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.features_layer(x)
        x = self.features_act(x)
        x = self.features_bn(x)

        x = self.output_layer(x)
        return x

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params
