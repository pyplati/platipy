import random

from torch.utils.data import BatchSampler
from torch.utils.data import Sampler


class ObserverSampler(Sampler):
    def __init__(self, data_source, num_observers):
        self.data_source = data_source
        self.num_observers = num_observers

    def __iter__(self):
        indices = list(range(int(len(self.data_source) / self.num_observers)))
        random.shuffle(indices)
        for i in indices:
            for o in range(self.num_observers):
                yield i * self.num_observers + o

    def __len__(self):
        return len(self.data_source)


print(
    list(
        BatchSampler(ObserverSampler(["x" for x in range(50)], 5), batch_size=10, drop_last=False)
    )
)
