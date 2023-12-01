import torch 
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func
from load_datasets import InstructionTuningDataset
from torch.utils.data import DataLoader, random_split 
import numpy as np
import tqdm 
import os 
import pickle as pkl
import random 
import time 
# from ds_config import ds_config
# import deepspeed 
import dao

class FeedForward(nn.Module): 
    def __init__(
        self,
        dim: int, 
        hdim: int 
        ):
        super().__init__() 
        self.c_fc = nn.Linear(dim, hdim, bias = True)
        self.c_proj = nn.Linear(hdim, dim, bias = True)
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

'''
flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1)):
"""dropout_p should be set to 0.0 during evaluation
If Q, K, V are already stacked into 1 tensor, this function will be faster than
calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
of the gradients of Q, K, V.
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.
Arguments:
    qkv: (batch_size, seqlen, 3, nheads, headdim)
    dropout_p: float. Dropout probability.
    softmax_scale: float. The scaling of QK^T before applying softmax.
        Default to 1 / sqrt(headdim).
    causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
    window_size: (left, right). If not (-1, -1), implements sliding window local attention.
Return:
    out: (batch_size, seqlen, nheads, headdim).
'''

class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_head: int,
        dropout = 0.1
        ):
        super().__init__()
        self.c_attn = nn.Linear(dim, 3*dim, bias=True)
        self.c_proj = nn.Linear(dim, dim, bias=True)
        self.n_head = n_head 
        self.dim = dim 
        self.dropout = dropout

    def forward(self, x):
        # x: (batch_size, seqlen, dim)
        qkv = self.c_attn(x)
        
        qkv = qkv.reshape(x.shape[0], x.shape[1], 3, self.n_head, -1)
        # qkv: (batch_size, seqlen, 3*dim)
        flash_qkv = qkv.to(torch.float16)
        dao.sync()
        out = flash_attn_qkvpacked_func(flash_qkv, self.dropout).reshape(x.shape[0], x.shape[1], -1).to(torch.float32)
        
        return self.c_proj(out)

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_head: int, 
        n_ctx: int,
        dropout = 0.1,
        dropout_attn = 0.1, 
        dropout_ff = 0.1
        ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_head, dropout)
        self.mlp = FeedForward(dim, dim*4)
        self.n_ctx = n_ctx 
        self.dropout_attn = dropout_attn 
        self.dropout_ff = dropout_ff
        self.n_forward = 0 
        self.n_forward_attn = 0
        self.n_forward_ff = 0
    def forward(self, x):
        # x: (batch_size, seqlen, dim)
        if torch.is_inference_mode_enabled():
            # x = x + self.n_forward_attn / self.n_forward * self.attn(self.ln_1(x))
            # x = x + self.n_forward_ff / self.n_forward * self.mlp(self.ln_2(x))
            x = x + (1 - self.dropout_attn) * self.attn(self.ln_1(x))
            x = x + (1 - self.dropout_ff) * self.mlp(self.ln_2(x))
            return x 
        
        self.n_forward += 1
        
        if random.random() > self.dropout_attn:
            x = x + self.attn(self.ln_1(x))
            self.n_forward_attn += 1
            
        if random.random() > self.dropout_ff:
            x = x + self.mlp(self.ln_2(x))
            self.n_forward_ff += 1
        
        return x

class gpt2(nn.Module):
    def __init__(
        self,
        n_vocab, 
        n_ctx, 
        n_head, 
        n_layer, 
        n_embd,
        dropout = 0.1, 
        dropout_attn = 0.1, 
        dropout_ff = 0.1 
    ):
        super().__init__()
        self.n_vocab, self.n_ctx, self.n_head, self.n_layer, self.dim = \
            n_vocab, n_ctx, n_head, n_layer, n_embd
        self.wte = torch.nn.Embedding(self.n_vocab, self.dim) 
        self.wpe = torch.nn.parameter.Parameter(torch.randn(self.n_ctx, self.dim), requires_grad=False)
        self.blocks = torch.nn.ModuleList([TransformerBlock(self.dim, self.n_head, self.n_ctx, dropout, dropout_attn, dropout_ff) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.dim)
        
    def forward(self, tokens):
        # tokens: (batch_size, seqlen)
        batch_size, seqlen = tokens.shape
        x = self.wte(tokens) + self.wpe[:seqlen, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x @ self.wte.weight.T

def _sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def compute_loss(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    # (batch, max_len, num_classes)
    log_probs = F.log_softmax(logits, dim=-1)
    # logits_flat: (batch * max_len, num_classes)
    dao.sync()
    log_probs_flat = log_probs.view(-1, log_probs.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    dao.sync()
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = _sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class Trainer:
    def __init__(self, args, models_dir = "models", dropout = 0.1):
        self.model_size = args.model_size 
        self.models_dir = models_dir
        self.dropout_attn = args.dropout_attn 
        self.dropout_ff = args.dropout_ff 
        self.batch_size = args.batch_size 
        self.use_deepspeed = args.use_deepspeed 
        self.profiling = args.profiling 
        self.use_tensorboard = args.use_tensorboard
        self.trace_prefix = args.trace_prefix 
        self.save_dir = ''
        folders = ['checkpoints', str(self.model_size), f'{self.dropout_attn}-{self.dropout_ff}', 'deepspeed' if self.use_deepspeed else 'torch']
        for folder in folders:
            self.save_dir = os.path.join(self.save_dir, folder)
            os.system(f'mkdir -p {self.save_dir}')
        
        from utils import load_encoder_hparams_and_params
        assert os.path.exists(os.path.join(self.models_dir, self.model_size))
        # params: K->V; K: parameter name; V: torch.tensor
        self.tokenizer, hparams, params = load_encoder_hparams_and_params(self.model_size, self.models_dir)
        self.n_vocab = self.tokenizer.n_vocab
        self.n_ctx = hparams['n_ctx']
        torch.set_default_dtype(torch.float32)
        
        self.model = gpt2(self.n_vocab, hparams['n_ctx'], hparams['n_head'], hparams['n_layer'], hparams['n_embd'],\
                    dropout, self.dropout_attn, self.dropout_ff)
        self.gen_and_load_state_dict(params)
        
        if self.use_deepspeed:
            self.model = deepspeed.zero.Init(module=self.model, remote_device='cpu')
        
        self.sos_id = self.tokenizer.get('<sos>')
        self.eos_id = self.tokenizer.get('<eos>')
        self.pad_id = self.tokenizer.get('<pad>')
        
        dataset = InstructionTuningDataset()
        # Created using indices from 0 to train_size.
        train_size = int(len(dataset) * 0.8)
        val_size = int(len(dataset) * 0.1)
        test_size = len(dataset) - train_size - val_size
        self.train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        self.test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, train_size + val_size + test_size))
        print('train: ', len(self.train_dataset), 'val:', len(self.val_dataset), 'test: ', len(self.test_dataset))
        
        if self.use_deepspeed:
            self.model_engine, _, _, _ = deepspeed.initialize(args=args, model=self.model, model_parameters=self.model.parameters(), config=ds_config)
            print('Deepspeed Enabled:', self.model_engine)

    def gen_model_dict(self, model_dict, data, names = []):
        if isinstance(data, dict):
            for k in data:
                names.append(k) 
                self.gen_model_dict(model_dict, data[k], names)
                names.pop()
        elif isinstance(data, list):
            for i, d in enumerate(data):
                names.append(str(i))
                self.gen_model_dict(model_dict, d, names)
                names.pop()
        elif isinstance(data, np.ndarray): 
            if names[-1] == 'b':
                names[-1] = 'bias'
            elif names[-1] == 'w' or names[-1] == 'g':
                names[-1] = 'weight'
                data = np.transpose(data)
            name = '.'.join(names)
            if name == 'wte': 
                name = 'wte.weight'
                assert self.n_vocab - data.shape[0] == 4
                n_vocab, hdim = data.shape 
                data = np.concatenate([data, np.random.normal(size=(self.n_vocab - n_vocab, hdim))], axis = 0)            
            model_dict[name] = torch.from_numpy(data)
        else:
            raise RuntimeError
        
    def gen_and_load_state_dict(self, params):
        model_dict = {}
        self.gen_model_dict(model_dict, params)
        names = set()
        for name, param in self.model.named_parameters():
            names.add(name)
        keys = set(model_dict.keys())
        assert keys == names
        self.model.load_state_dict(model_dict)
    
    def prepare_data(self, data):
        tokens = [self.tokenizer.encode(x)[:self.n_ctx-1] for x in data]
        max_len = max([len(x) for x in tokens])
        # (batch_size, max_len)
        input_tokens = torch.tensor([[self.sos_id] + token + (max_len - len(token)) * [self.pad_id] for token in tokens], dtype=torch.long).to('cuda:0')
        # (batch_size, max_len)
        label_tokens = torch.tensor([token + [self.eos_id] + (max_len - len(token)) * [self.pad_id] for token in tokens], dtype=torch.long).to('cuda:0')
        # (batch_size)
        length = torch.tensor([len(x) + 1 for x in tokens], dtype=torch.long).to('cuda:0')
        assert max_len < self.n_ctx
        
        return input_tokens, label_tokens, length
    
    def train(self, epoch):        
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.to('cuda:0') 
            
        training_losses = []
        eval_losses = []
        
        # for profiling 
        if self.profiling: 
            os.system('mkdir -p traces')
            if self.trace_prefix is not None: 
                prefix = self.trace_prefix
            else:
                prefix = "deepspeed-" if self.use_deepspeed else "torch-"
            trace_dir = f'./traces/{prefix}'
            os.system(f'mkdir -p {trace_dir}')
            def trace_handler(p):
                output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
                print(output)
                
                p.export_chrome_trace(f"./{trace_dir}/{prefix}" + str(p.step_num) + ".json")
            if self.use_tensorboard:
                trace_handler = torch.profiler.tensorboard_trace_handler(trace_dir)
                    
            with torch.profiler.profile(
                schedule = torch.profiler.schedule(
                    wait = 1,
                    warmup = 2, 
                    active = 2,
                    repeat = 1),
                on_trace_ready = trace_handler,
                with_stack = True,
                profile_memory=True if self.use_tensorboard else False,
            ) as profiler: 
                for j, (data, label) in tqdm.tqdm(enumerate(train_loader)):
                    input_tokens, label_tokens, length = self.prepare_data(data)
                    if self.use_deepspeed:
                        logits = self.model_engine(input_tokens).view(-1, self.n_vocab)
                        loss = compute_loss(logits, label_tokens, length)
                        self.model_engine.backward(loss)
                        self.model_engine.step()
                    else:
                        opt.zero_grad()
                        logits = self.model(input_tokens).view(-1, self.n_vocab)
                        loss = compute_loss(logits, label_tokens, length)
                        loss.backward()
                        opt.step()
                    profiler.step()
                    if j >= 5: break
            return 
        
        t0 = time.time()
        for i in range(epoch):
            for j, (data, label) in tqdm.tqdm(enumerate(train_loader)):
                input_tokens, label_tokens, length = self.prepare_data(data)
                
                if self.use_deepspeed:
                    logits = self.model_engine(input_tokens).view(-1, self.n_vocab)
                    loss = compute_loss(logits, label_tokens, length)
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                else:
                    opt.zero_grad()
                    dao.sync() ##
                    logits = self.model(input_tokens).view(-1, self.n_vocab)
                    loss = compute_loss(logits, label_tokens, length)
                    dao.sync()
                    loss.backward()
                    dao.sync() ##
                    opt.step()
                    
                dao.sync()
                loss_val = loss.item()
                print(i, j, 'loss:', loss_val)
                training_losses.append((time.time() - t0, loss_val))

                start = time.time()
                if (j+1)%1000 == 0:
                    torch.save(self.model.state_dict(), f'{self.save_dir}/{self.model_size}_epoch_{i}_iter_{j+1}.pt')
                    with open(f'{self.save_dir}/training_losses.pkl', 'wb') as f:
                        pkl.dump(training_losses, f)
                    with torch.inference_mode(True):
                        val_loss = 0
                        n = 0
                        for k, (data, label) in tqdm.tqdm(enumerate(val_loader)):
                            input_tokens, label_tokens, length = self.prepare_data(data)
                            logits = self.model(input_tokens).view(-1, self.n_vocab)
                            tensor_loss = compute_loss(logits, label_tokens, length)
                            dao.sync()
                            val_loss += tensor_loss.item()                            
                            n += 1 
                        print(f"epoch {i}, iter {j+1}, train loss {loss.item()}, val loss {val_loss/n}")
                    eval_losses.append(val_loss/n)
                    with open(f'{self.save_dir}/eval_losses.pkl', 'wb') as f:
                        pkl.dump(eval_losses, f)
                t0 += time.time() - start 

    def eval(self):
        with open(f'{self.save_dir}/training_losses.pkl', 'rb') as f:
            training_losses = pkl.load(f)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size*2, shuffle=False)
        test_losses = []
        for filename in os.listdir(self.save_dir):
            if '.pt' not in filename: continue 
            _,_,epoch,_,iter=filename.strip('.pt').split('_')
            epoch, iter = int(epoch), int(iter)
            elapsed_time, training_loss = training_losses[epoch*5200 + iter]
            self.model.load_state_dict(torch.load(os.path.join(self.save_dir, filename)))
            self.model.to('cuda:0') 
            with torch.inference_mode(True):
                test_loss = 0
                n = 0
                for k, (data, label) in tqdm.tqdm(enumerate(test_loader)):
                    input_tokens, label_tokens, length = self.prepare_data(data)
                    logits = self.model(input_tokens).view(-1, self.n_vocab)
                    tensor_loss = compute_loss(logits, label_tokens, length)
                    dao.sync()
                    test_loss += tensor_loss.item()
                    n += 1 
                test_loss /= n 
                print(f"test loss {test_loss}")
                test_losses.append((elapsed_time, training_loss, test_loss))
        with open(f'{self.save_dir}/test_losses.pkl', 'wb') as f:
            pkl.dump(test_losses, f)
    
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='124M')
parser.add_argument('--dropout_attn', type=float, default=0)
parser.add_argument('--dropout_ff', type=float, default=0)
parser.add_argument('--batch_size', type=int, default = 8)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--use_deepspeed', action='store_true')
parser.add_argument('--profiling', action='store_true')
parser.add_argument('--use_tensorboard', action='store_true')
parser.add_argument('--trace_prefix', type=str, default=None)
parser.add_argument('--use_lora', action='store_true')
# parser = deepspeed.add_config_arguments(parser)

if __name__ == '__main__':
    dao.verbose(0)
    dao.launch()
    # for model_size in ['124M', '355M', '774M', '1558M']:
    # for model_size in ['355M']:
    #     for dropout_attn in [0, 0.1, 0.3, 0.5]:
    #         for dropout_ff in [0, 0.1, 0.3, 0.5]:
    args = parser.parse_args()
    # if args.model_size == '124M': batch_size = 8
    # else: batch_size = 1     
    trainer = Trainer(args, models_dir = 'models')
    if args.eval:
        trainer.eval()
    else:
        trainer.train(1)
        
# python ./gpt2_torch_dao.py --model_size 124M --batch_size 2 --eval
# python ./gpt2_torch_dao.py --model_size 124M --batch_size 2

'''
baseline: 
    W: [1024,1024], trainable
    X = WX + B;
PEFT: Parameter Efficient Fine Tuning;
- lora:
    W: [1024,1024], pretrained, not trainable
    P: [1024, 4], trainable 
    Q: [4, 1024], trainable 
    
    W'=W+P@Q
    X = activation(W'X + B);
- Adaptor: 
    x = input()
    for l in layers: 
        x = l(x) # l not trainable
        x = g(x) # g trainable, e.g. g = nn.Linear(1024, 1024)
    return output(x) 
- implementation: peft library;
'''