import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import csv

print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#globals from gpt.py
USE_RESIDUAL_ATTN = True
USE_RESIDUAL_FFN  = True
USE_PRENORM       = True

block_size = 256
n_embd = 128
n_head = 4
n_layer = 3
dropout = 0.3
vocab_size = 0 

torch.manual_seed(1337)

#takem from gpt.py

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        if USE_PRENORM:
            #Pre-LN
            attn_in  = self.ln1(x)
            attn_out = self.sa(attn_in)
            x = x + attn_out if USE_RESIDUAL_ATTN else attn_out

            ffn_in  = self.ln2(x)
            ffn_out = self.ffwd(ffn_in)
            x = x + ffn_out if USE_RESIDUAL_FFN else ffn_out
        else:
            #post-LN
            attn_out = self.sa(x)
            x = x + attn_out if USE_RESIDUAL_ATTN else attn_out
            x = self.ln1(x)

            ffn_out = self.ffwd(x)
            x = x + ffn_out if USE_RESIDUAL_FFN else ffn_out
            x = self.ln2(x)
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

#load checkpoint from model 1 and get hyperparams

ckpt_path = "model1_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
print(f"loaded checkpoint model {ckpt_path}")

cfg = ckpt["config"]

#overwrite the training configs to match the loaded model
USE_RESIDUAL_ATTN = cfg["USE_RESIDUAL_ATTN"]
USE_RESIDUAL_FFN = cfg["USE_RESIDUAL_FFN"]
USE_PRENORM = cfg["USE_PRENORM"]
vocab_size = cfg["vocab_size"]
block_size= cfg["block_size"]
n_embd = cfg["n_embd"]
n_head = cfg["n_head"]
n_layer = cfg["n_layer"]
dropout = cfg["dropout"]

chars = ckpt["chars"]

#encoder/decoder from saved chars
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join(itos[i] for i in indices)

#istantiate model with correct configs
model = GPTLanguageModel().to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()
print(sum(p.numel() for p in model.parameters())/1e6, "M parameters")

#load test texts
with open('input_childSpeech_testSet.txt', 'r', encoding='utf-8') as f:
    child_test_text = f.read()

with open('input_shakespeare.txt', 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()


#evaluation helper
@torch.no_grad()
def eval_loss_on_text(raw_text, label="", eval_iters=200, batch_size=64):
    ##estimate average loss and ppl on text
    data = torch.tensor(encode(raw_text), dtype=torch.long)
    
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(0, len(data)-block_size-1, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    print(f"[{label}] loss: {avg_loss:.4f}, ppl: {ppl:.2f}")
    return avg_loss, ppl


results = []

child_loss, child_ppl = eval_loss_on_text(child_test_text, label="child_test")
results.append(("child_test", child_loss, child_ppl))

shak_loss, shak_ppl = eval_loss_on_text(shakespeare_text, label="shakespeare")
results.append(("shakespeare", shak_loss, shak_ppl))

##save results to csv
#with open("gpt_eval_results_model1.csv", "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow(["dataset", "loss", "ppl"])
#    for name, loss, ppl in results:
#        writer.writerow([name, loss, ppl])
#
#print("Saved evaluation results -> gpt_eval_results_model1.csv")


#get performance on both datasets with random weight model to compare 

#load settings from checkpoint from model 1 to get hyperparams to keep them constant between 2 mdoels
ckpt_path = "model1_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
print(f"loaded checkpoint model {ckpt_path}")

cfg = ckpt["config"]

#overwrite the training configs to match the loaded model
USE_RESIDUAL_ATTN = cfg["USE_RESIDUAL_ATTN"]
USE_RESIDUAL_FFN = cfg["USE_RESIDUAL_FFN"]
USE_PRENORM = cfg["USE_PRENORM"]
vocab_size = cfg["vocab_size"]
block_size = cfg["block_size"]
n_embd = cfg["n_embd"]
n_head = cfg["n_head"]
n_layer = cfg["n_layer"]
dropout = cfg["dropout"]


#reconstrcut vocab from saved chars
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join(itos[i] for i in indices)

#load tests set
with open('input_childSpeech_testSet.txt', 'r', encoding='utf-8') as f:
    child_test_text = f.read()

with open('input_shakespeare.txt', 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()

#eval loss and ppl
#take model and raw text as input

@torch.no_grad()
def eval_loss_on_text(model, raw_text, label="", eval_iters=200, batch_size=64):
    model.eval()
    #encode raw text to tensor
    data = torch.tensor(encode(raw_text), dtype=torch.long)
    losses = []
    #for every batch get random sample and record cross entropy loss
    for _ in range(eval_iters):
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    print(f"[{label}] loss: {avg_loss:.4f}, ppl: {ppl:.2f}")
    return avg_loss, ppl


#1) Random-weight model

torch.manual_seed(1337)  
model_rand = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model_rand.parameters())/1e6, "M parameters (random model)")

rand_child_loss, rand_child_ppl = eval_loss_on_text(model_rand, child_test_text, label="RANDOM_child_test"
)
rand_shake_loss, rand_shake_ppl = eval_loss_on_text(model_rand, shakespeare_text, label="RANDOM_shakespeare"
)


#2) Trained model (Model 1)
model_trained = GPTLanguageModel().to(device)
model_trained.load_state_dict(ckpt["state_dict"])
print(sum(p.numel() for p in model_trained.parameters())/1e6, "M parameters (trained model)")

child_loss, child_ppl = eval_loss_on_text(
    model_trained, child_test_text, label="TRAINED_child_test"
)
shake_loss, shake_ppl = eval_loss_on_text(
    model_trained, shakespeare_text, label="TRAINED_shakespeare"
)

#save to csv
results = [
    ("random","child_test",rand_child_loss,rand_child_ppl),
    ("random","shakespeare",rand_shake_loss,rand_shake_ppl),
    ("trained","child_test",child_loss,child_ppl),
    ("trained","shakespeare",shake_loss,shake_ppl),
]

with open("gpt_eval_results_model1_random_vs_trained.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model_type", "dataset", "loss", "ppl"])
    for row in results:
        writer.writerow(row)

print("Saved evaluation results gpt_eval_results_model1_random_vs_trained.csv")
