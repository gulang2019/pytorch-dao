import torch as t
import torch._C as dao
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from time import sleep


def test_1d_model(dim=3):
    model = nn.Sequential(nn.Linear(dim, 1)) # nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
    dao_model = deepcopy(model).to('cuda:0')
    loss_fn = nn.MSELoss()

    a = t.rand(1, dim)
    actual = a.sum(-1, keepdim=True) # shape=(1, 1)
    pred = model(a)
    loss = loss_fn(pred, actual)
    # Ground Truth: torch_cpu
    print(f"GT: pred {pred}; loss {loss}")
    
    # Unit under test
    a = a.to('cuda:0')
    actual = actual.to('cuda:0')
    dao_pred = dao_model(a)
    # dao_loss = loss_fn(dao_pred, actual)
    dao.sync()
    sleep(1)
    print(f"DAO: pred {dao_pred}; loss {None}")


def test_seq_model(seq=4, dim=3):
    pass


if __name__ == '__main__':
    dao.verbose(1)
    dao.launch()
    t.manual_seed(0)
    test_1d_model()
