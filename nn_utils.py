import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(outer_dim),
            nn.PReLU(num_parameters=outer_dim),
            nn.Conv2d(outer_dim, inner_dim, 1),
            nn.BatchNorm2d(inner_dim),
            nn.PReLU(num_parameters=inner_dim),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.PReLU(num_parameters=inner_dim),
            nn.Conv2d(inner_dim, outer_dim, 1),
        )

    def forward(self, input):
        return input + self.net(input)

class SkipConnection(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.inner_net = nn.Sequential(*args)

    def forward(self, input):
        return input + self.inner_net(input)

class MemoryLayer(nn.Module):
    storage = {}

    def __init__(self, id, output=False, add=False):
        super().__init__()
        self.id = id
        self.output = output
        self.add = add

    def forward(self, input):
        if not self.output:
            self.storage[self.id] = input
            return input
        else:
            if self.id not in self.storage:
                err = 'MemoryLayer: id \'%s\' is not initialized. '
                err += 'You must execute MemoryLayer with the same id '
                err += 'and output=False before this layer.'
                raise ValueError(err)
            stored = self.storage[self.id]
            if not self.add:
                data = torch.cat([input, stored], 1)
            else:
                data = input + stored
            return data
