import torch.nn.functional as F
import torch

import utils

class Aggregator(torch.nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(utils.CNT_CATS * utils.TOP_K_PROP_NAMES, utils.AGG_HIDDEN_SIZE, device = utils.DEVICE),
            torch.nn.ReLU(),
            torch.nn.Linear(utils.AGG_HIDDEN_SIZE, utils.CNT_CATS * utils.TOP_K_PROP_NAMES, device = utils.DEVICE),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.tensor):
        return self.seq(x)


class Updater(torch.nn.Module):
    def __init__(self):
        super(Updater, self).__init__()

        self.cell = torch.nn.GRU(input_size = utils.CNT_CATS * utils.TOP_K_PROP_NAMES, hidden_size = utils.UPD_HIDDEN_SIZE, device = utils.DEVICE)
        self.linear_expand = torch.nn.Linear(utils.UPD_HIDDEN_SIZE, utils.CNT_CATS * utils.TOP_K_PROP_NAMES, device = utils.DEVICE)

    def forward(self, embedding: torch.tensor):
        next_embedding, next_hidden = self.cell(embedding.unsqueeze(dim = 0))

        return self.linear_expand(next_embedding[0])


class GlobalAttnPool(torch.nn.Module):
    def __init__(self):
        super(GlobalAttnPool, self).__init__()

        self.linear = torch.nn.Linear(utils.CNT_CATS * utils.TOP_K_PROP_NAMES, 1, device = utils.DEVICE)

    def forward(self, x: torch.tensor):
        # x.shape = [node_count, embedding_size].
        
        logits = F.softmax(self.linear(x), dim = 0) # logits.shape = [node_count, 1]
        mean_x = (x * logits).sum(dim = 0) # shape = [embedding_size].

        return mean_x


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(utils.CNT_CATS * utils.TOP_K_PROP_NAMES, utils.AGG_HIDDEN_SIZE, device = utils.DEVICE),
            torch.nn.ReLU(),
            torch.nn.Linear(utils.AGG_HIDDEN_SIZE, 1, device = utils.DEVICE),
            torch.nn.Sigmoid()
        )

    def forward(self, x: torch.tensor):
        return self.seq(x)
