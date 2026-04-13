import math
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, confusion_matrix
import random
import re

#script toggles for training and testing
#TRAIN False if only testing a pre-trained model
#TEST False if only training without testing
#BOTH TRUE to do both training and testing at the same time
TRAIN=False
TEST=True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = BASE_DIR
MODEL_DIR = BASE_DIR           # relative to current working dir
os.makedirs(MODEL_DIR, exist_ok=True) # create it if it doesn't exist

MODEL_BASENAME = "model_weights_part2"

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_BASENAME)
MODEL_BEST_PATH = os.path.join(MODEL_DIR, MODEL_BASENAME + ".pth")
MODEL_LAST_PATH = os.path.join(MODEL_DIR, MODEL_BASENAME + "_last.pth")

TRAIN_PATH = os.path.join(DATASET_DIR, "bool_4input_train.txt")
TEST_PATH  = os.path.join(DATASET_DIR, "bool_4input_test.tsv")
# hyperparameters
batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 8000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 96
n_head = 6
n_layer = 6
dropout = 0.1
# ------------

#helper functions for alibi slope and bias

def get_alibi_slopes(n_heads):
    def slopes_power_2(n):
        start = 2**(-2**-(math.log2(n)-3))
        ratio = start
        return [start*ratio**i for i in range(n)]
    if math.log2(n_heads).is_integer():
        return slopes_power_2(n_heads)
    else:
        #return the closest power of 2
        closest = 2**math.floor(math.log2(n_heads))
        slopes = slopes_power_2(closest)
        extra_slopes = get_alibi_slopes(2*closest)
        slopes.extend(extra_slopes[0::2][:n_heads - closest])
        return slopes
    
# function to get the alibi bias tensor
def get_alibi_bias(block_size, slope, device= None):
    i = torch.arange(block_size, device=device).unsqueeze(0)
    j = torch.arange(block_size, device=device).unsqueeze(1)
    relative_positions = (j - i).clamp(min=0)  #get causal distance
    return (-slope * relative_positions)

torch.manual_seed(1337)

#custom token list 
TOKEN_LIST = re.compile(r"XNOR|NAND|NOR|XOR|AND|OR|NOT|[01]|[()=]")

def tokenize_bool(line):
    line = line.strip()
    tokens = TOKEN_LIST.findall(line)
    if not tokens or "=" not in tokens:
        return None
    return tokens

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#point to the mathGPT train set
with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
#vocab_size = len(chars)
# create a mapping from characters to integers
#stoi = { ch:i for i,ch in enumerate(chars) }
#itos = { i:ch for i,ch in enumerate(chars) }
#encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
#decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

#Nnew tokeniser for boolean equations
lines = [ln.strip("\n") for ln in text.split("\n") if ln.strip()]

token_lines = []
for ln in lines:
    tokens = tokenize_bool(ln)
    if tokens is not None:
        token_lines.append(tokens)

PAD = "<PAD>"
EOL = "<EOL>"

vocab = sorted(set(token for tokens in token_lines for token in tokens) | {PAD, EOL})
vocab_size = len(vocab)

stoi = {token:i for i, token in enumerate(vocab)}
itos = {i:token for token, i in stoi.items()}

def encode_tokens(toks):
    return [stoi[t] for t in toks]

def decode_tokens(ids):
    return [itos[i] for i in ids]


#constants for roc curves
ID0 = stoi["0"]
ID1 = stoi["1"]
ROC_DIR = os.path.join(MODEL_DIR, "roc_curves/" + MODEL_BASENAME)
os.makedirs(ROC_DIR, exist_ok=True)

#inplement line based dataset - instead of large text vomit
random.seed(1337)
#random.shuffle(lines)
split_idx = int(0.8 * len(lines))
#train_lines = lines[:split_idx]
#val_lines = lines[split_idx:]
#val_lines = 0 * lines #use all data for training
random.seed(1337)
random.shuffle(token_lines)
split_idx = int(0.8 * len(token_lines))
#new function to compute split after shuffling - all basic ops in train and then multi op patterns can be in train and val
def compute_split(token_lines, split_ratio=0.8):
    train_lines = []
    val_lines = []
    basic_ops = {"AND", "OR", "NAND", "NOR", "NOT", "XOR", "XNOR"}
    for toks in token_lines:
        ops_in_exp = set(t for t in toks if t in basic_ops)
        if len(ops_in_exp) == 1:
            #if only one op send to train
            train_lines.append(toks)
        else: #normal split according to split ratio
            if len(train_lines) / len(token_lines) < split_ratio:
                train_lines.append(toks)
            else:
                val_lines.append(toks)
    return train_lines, val_lines

#train_token_lines = token_lines[:split_idx]
#val_token_lines   = token_lines[split_idx:]

#call split function 
train_token_lines, val_token_lines = compute_split(token_lines, split_ratio=0.8)


max_line_length = max(len(toks) + 1 for toks in token_lines) 
print("max line lenght, " , max_line_length)
print("block size ", block_size)

OPS = ["AND", "OR", "NAND", "NOR", "NOT", "XOR", "XNOR", "MULTI_4_INPUT", "MULTI_6_INPUT", 
       "HOLDOUT_4_INPUT", "UNSEEN_4", "UNSEEN_6", "MULTI_3_INPUT", "HOLDOUT_3_INPUT"]

def detect_ops(toks):
    #count inputs before =
    eq_i = toks.index("=")
    n_inputs = sum(1 for t in toks[:eq_i] if t in ("0", "1"))
    if n_inputs == 4:
        return "MULTI_4_INPUT"
    if n_inputs == 6:
        return "MULTI_6_INPUT"
    if n_inputs == 3:
        return "MULTI_3_INPUT"
    

    # unary-only NOT (like NOT(0)=1)
    if toks[0] == "NOT":
        return "NOT"

    # basic binary gates (look for operator token)
    if "XNOR" in toks: return "XNOR"
    if "XOR"  in toks: return "XOR"
    if "NAND" in toks: return "NAND"
    if "NOR"  in toks: return "NOR"
    if "AND"  in toks: return "AND"
    if "OR"   in toks: return "OR"
    return None



def build_encoded_by_op(token_lines_list):
    by_op = {op: [] for op in OPS}
    for toks in token_lines_list:
        op = detect_ops(toks)
        if op is None:
            continue
        ids = encode_tokens(toks + [EOL])
        by_op[op].append(torch.tensor(ids, dtype=torch.long))
    return by_op


encoded_train_by_op = build_encoded_by_op(train_token_lines)
encoded_val_by_op   = build_encoded_by_op(val_token_lines)

print("Train per-op counts:", {op: len(v) for op, v in encoded_train_by_op.items() if len(v) > 0})
print("Val per-op counts:",   {op: len(v) for op, v in encoded_val_by_op.items() if len(v) > 0})

def get_batch_lines(split):
    by_ops = encoded_train_by_op if split == 'train' else encoded_val_by_op
    active_ops = [op for op in OPS if len(by_ops[op]) > 0]

    #proportional sampling based on available samples per operation
    counts = np.array([len(by_ops[op]) for op in active_ops], dtype=np.float64)
    probs = counts / counts.sum()

    selected_lines = []
    for _ in range(batch_size):
        op = np.random.choice(active_ops, p=probs)
        selected_lines.append(random.choice(by_ops[op]))

    PAD_ID = stoi[PAD]
    xb = torch.full((batch_size, block_size), PAD_ID, dtype=torch.long)
    yb = torch.full((batch_size, block_size), PAD_ID, dtype=torch.long)

    for i, seq in enumerate(selected_lines):
        seq = seq[:block_size + 1]
        if seq.numel() < block_size + 1:
            pad = torch.full((block_size + 1 - seq.numel(),), PAD_ID, dtype=torch.long)
            seq = torch.cat([seq, pad], dim=0)
        xb[i] = seq[:-1]
        yb[i] = seq[1:]

    #only train on tokens after =
    EQ_ID = stoi["="]
    IGNORE = -9
    yb_hidden = torch.full_like(yb, IGNORE)

    for i in range(batch_size):
        eq_pos = (xb[i] == EQ_ID).nonzero(as_tuple=True)[0]
        if eq_pos.numel() == 0:
            continue
        x = int(eq_pos[0].item())
        yb_hidden[i, x] = yb[i, x]

    return xb.to(device), yb_hidden.to(device)

                    
    

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_lines(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, alibi_slope):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        
        #new for alibi 
        self.alibi_slope = alibi_slope  # single head, get single slope
        self.register_buffer('alibi_bias', get_alibi_bias(block_size, alibi_slope, device=None))  # (block_size, block_size)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        #dot product already scaled so just add bias
        wei = wei + self.alibi_bias[:T, :T]  # (B, T, T)
        
        
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
        
        #alibi slopes for each head
        slopes = get_alibi_slopes(num_heads)
        
        
        self.heads = nn.ModuleList([Head(head_size, alibi_slope = slopes[i]) for i in range(num_heads)])
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
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #self.position_embedding_table = nn.Embedding(block_size, n_embd)
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
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        #x = tok_emb + pos_emb # (B,T,C)
        #x = self.blocks(x) # (B,T,C)
        #x = self.ln_f(x) # (B,T,C)
        #tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb # (B,T,C)
        x= self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-9)

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
    
    def generate_greedy(self,idx,max_new_tokens):
        """version of generate that uses greedy sampling - always picks the most probable next token - /n
        this is more applicable to maths problems where there is a single correct answer"""
        
        # idx is (B, T) array of indices in the current context
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        # get the predictions
        logits, _ = self(idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        # sample from the distribution
        allowed = [stoi["0"], stoi["1"]]
        mask = torch.full_like(logits, float("-inf"))
        mask[:, allowed] = 0
        logits = logits + mask
        idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    def detokenize(tokens):
        #join tokens into a readable string to output at the end
        #take existing prompt from test and append generated tokens
        out = []
        for t in tokens:
            #if the token is a binary op append it with spaces like in the dataset
            #treat NOT different as it has a different spacing style
            if t in ("AND", "OR", "NAND", "NOR", "XOR", "XNOR"):
                out.append(" " + t + " ")
            elif t == "NOT":
                out.append("NOT")
            elif t in ("(", ")", "="):
                out.append(t)
            elif t in ("0", "1"):
                out.append(t)
                #doesnt matter to include these they just make things messy
            elif t in (EOL, PAD):
                continue
            else:
                out.append(t) #unknown token just append as is for now
            equation_combined = ''.join(out) #
        return equation_combined
            

model = GPTLanguageModel().to(device)
#m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
m = model
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

#training data logs
train_losses = []
val_losses = []
step_list = []
log_train_ppl,  log_val_ppl  = [], []

# best model tracking
best_val_loss = float('inf')
best_iter = -1

if TRAIN:
  # training loop
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            
            #get train and val loss and perplexity for each iteration
            train_loss = float(losses['train'])
            val_loss = float(losses['val'])
            train_ppl = math.exp(train_loss)
            val_ppl = math.exp(val_loss)
            
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            #append to logs
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            step_list.append(iter)
            log_train_ppl.append(train_ppl);   
            log_val_ppl.append(val_ppl)
            
            #save best model and lowest val model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iter = iter
                torch.save(m.state_dict(), MODEL_BEST_PATH)
                print("saved best model to ", MODEL_BEST_PATH)
        # sample a batch of data
        xb, yb = get_batch_lines('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    # save last model after training
    torch.save(model.state_dict(), MODEL_LAST_PATH)
    print(f"Training done. Best val loss {best_val_loss:.4f} at step {best_iter}.")
    print(f"Saved last model to {MODEL_LAST_PATH}")
    
if TRAIN:    
    with open(os.path.join(MODEL_DIR, MODEL_BASENAME + '_train_losses.csv'),
              mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'train_loss', 'val_loss', 'train_ppl', 'val_ppl'])
        for step, train_loss, val_loss, trn_ppl, vp in zip(step_list, train_losses, val_losses, log_train_ppl, log_val_ppl):
            writer.writerow([step, train_loss, val_loss, trn_ppl, vp])

# generate from the model
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

#test area - test is in tsv format as prompt/answer/operation
#this allows for calculation accuract in total and per operation

#load the testset
def load_testset(test_path):
    test_rows = []
    with open(test_path, "r",encoding="utf-8") as f:
        reader = csv.reader(f,delimiter="\t")
        header = next(reader)
        for row in reader:
            prompt = row[0]
            answer = row[1]
            operation = row[2]
            test_rows.append((prompt,answer,operation))
    return test_rows

#fucntion for getting probability of next token being '1' for roc curve plotting
@torch.no_grad()
def probs_one(model,prompt):
    model.eval()
    tokens = tokenize_bool(prompt)
    idx = torch.tensor([encode_tokens(tokens)], dtype=torch.long, device=device)
    logits, _ = model(idx)
    next_token_logits = logits[:,-1,:]
    
    #restruct to only 1 or 0
    allowed = next_token_logits[:, [ID0, ID1]]
    probs = F.softmax(allowed, dim=-1)[0,1].item() #probability of '1'
    model.train()
    return probs

@torch.no_grad()
@torch.no_grad()
def plot_roc_curve(model, test_rows):
    """plot all three test expressions on singular plot to save room"""
    model.eval()

    ops_to_plot = ["HOLDOUT_3_INPUT", "HOLDOUT_4_INPUT", "UNSEEN_4"]
    operation_data = {op: ([], []) for op in ops_to_plot}  

    #collect labels and scores
    for prompt, answer, operation in test_rows:
        if operation not in operation_data:
            continue
        prob_one = probs_one(model, prompt)
        true_label = 1 if answer == "1" else 0
        operation_data[operation][0].append(true_label)
        operation_data[operation][1].append(prob_one)

    plt.figure(figsize=(7.5, 6))

    for operation in ops_to_plot:
        true_labels, prob_scores = operation_data[operation]
        fpr, tpr, _ = roc_curve(true_labels, prob_scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f"{operation} (AUC={roc_auc:.2f})")
        any_plotted = True

    # diagonal baseline
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random (AUC=0.50)")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Holdout/Unseen Sets)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(ROC_DIR, "roc_curves_holdout3_holdout4_unseen4.png")
    
    plt.savefig(out_path, dpi=200)
        

    plt.close()
    model.train()

@torch.no_grad()
def test_model(model,test_rows,max_new_tokens=1):
    model.eval()
    total = 0
    correct = 0
    operation_total = {"MULTI_4_INPUT":0, "HOLDOUT_4_INPUT":0, "MULTI_6_INPUT":0, "UNSEEN_4":0, "UNSEEN_6":0, "MULTI_3_INPUT":0, "HOLDOUT_3_INPUT":0}
    operation_correct = {"MULTI_4_INPUT":0, "HOLDOUT_4_INPUT":0, "MULTI_6_INPUT":0, "UNSEEN_4":0, "UNSEEN_6":0, "MULTI_3_INPUT":0, "HOLDOUT_3_INPUT":0}

    for prompt, answer, operation in test_rows:
        
        prompt_toks = tokenize_bool(prompt)
        idx = torch.tensor([encode_tokens(prompt_toks)], dtype=torch.long, device=device)

        output_idx = model.generate_greedy(idx, max_new_tokens=max_new_tokens)
        out_toks = decode_tokens(output_idx[0].tolist())

        # prediction is the NEXT token after the prompt
        pred_tok = out_toks[len(prompt_toks)]
        pred_answer = pred_tok
        
        
        #extract pred answer
        
        is_correct = (pred_answer == answer)
        total += 1
        correct += int(is_correct)
        
        if operation in operation_total:
            operation_total[operation] += 1
            operation_correct[operation] += int(is_correct)
            
    #overall_acc = correct / total * 100
    #print(f"Overall Test Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    
    overall_acc = correct / total if total > 0 else 0.0
    MULTI_4_INPUT_acc = (
        operation_correct["MULTI_4_INPUT"] / operation_total["MULTI_4_INPUT"]
        if operation_total["MULTI_4_INPUT"] > 0 else 0.0
    )
    MULTI_6_INPUT_acc = (
        operation_correct["MULTI_6_INPUT"] / operation_total["MULTI_6_INPUT"]
        if operation_total["MULTI_6_INPUT"] > 0 else 0.0
    )
    UNSEEN_4_acc = (
        operation_correct["UNSEEN_4"] / operation_total["UNSEEN_4"]
        if operation_total["UNSEEN_4"] > 0 else 0.0
    )
    UNSEEN_6_acc = (
        operation_correct["UNSEEN_6"] / operation_total["UNSEEN_6"]
        if operation_total["UNSEEN_6"] > 0 else 0.0
    )
    HOLDOUT_4_INPUT_acc = (
        operation_correct["HOLDOUT_4_INPUT"] / operation_total["HOLDOUT_4_INPUT"]
        if operation_total["HOLDOUT_4_INPUT"] > 0 else 0.0
    )
    MULTI_3_INPUT_acc = (
        operation_correct["MULTI_3_INPUT"] / operation_total["MULTI_3_INPUT"]
        if operation_total["MULTI_3_INPUT"] > 0 else 0.0
    )
    HOLDOUT_3_INPUT_acc = (
        operation_correct["HOLDOUT_3_INPUT"] / operation_total["HOLDOUT_3_INPUT"]
        if operation_total["HOLDOUT_3_INPUT"] > 0 else 0.0
    )
        
    print("\n=== BoolGPT test results ===")
    print(f"Overall equation accuracy: {overall_acc:.4f}")
    #print(f"Accuracy for 'AND':          {AND_acc:.4f}")
    #print(f"Accuracy for 'OR':          {OR_acc:.4f}\n")
    #print(f"Accuracy for 'NAND':        {NAND_acc:.4f}")
    #print(f"Accuracy for 'NOR':         {NOR_acc:.4f}\n")
    #print(f"Accuracy for 'XOR':         {XOR_acc:.4f}")
    #print(f"Accuracy for 'XNOR':        {XNOR_acc:.4f}\n")
    #print(f"Accuracy for 'NOT':         {NOT_acc:.4f}")
    print(f"Accuracy for 'MULTI_4_INPUT':       {MULTI_4_INPUT_acc:.4f}")
    print(f"Accuracy for 'HOLDOUT_4_INPUT':       {HOLDOUT_4_INPUT_acc:.4f}")
    print(f"Accuracy for 'MULTI_6_INPUT':       {MULTI_6_INPUT_acc:.4f}")
    print(f"Accuracy for 'UNSEEN_4':       {UNSEEN_4_acc:.4f}")
    print(f"Accuracy for 'UNSEEN_6':       {UNSEEN_6_acc:.4f}")
    print(f"Accuracy for 'MULTI_3_INPUT':       {MULTI_3_INPUT_acc:.4f}")
    print(f"Accuracy for 'HOLDOUT_3_INPUT':       {HOLDOUT_3_INPUT_acc:.4f}\n")

    model.train()
    return overall_acc, {"MULTI_4_INPUT": MULTI_4_INPUT_acc, "HOLDOUT_4_INPUT": HOLDOUT_4_INPUT_acc, "MULTI_6_INPUT": MULTI_6_INPUT_acc, "UNSEEN_4": UNSEEN_4_acc, "UNSEEN_6": UNSEEN_6_acc, "MULTI_3_INPUT": MULTI_3_INPUT_acc, "HOLDOUT_3_INPUT": HOLDOUT_3_INPUT_acc}   
if TEST:
    # always evaluate best checkpoint
    state_dict = torch.load(MODEL_BEST_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("loaded model from", MODEL_BEST_PATH)
    
    test_rows = load_testset(TEST_PATH)
    print(f"Loaded {len(test_rows)} test examples from {TEST_PATH}")
    
    overall_acc, op_acc = test_model(model, test_rows, max_new_tokens=1)
    
    plot_roc_curve(model, test_rows)
    print("Saved ROC curves to:", ROC_DIR)

    with open(os.path.join(MODEL_DIR, MODEL_BASENAME + "_test_results.csv"),
              mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
        "overall_acc",
        "MULTI_4_INPUT_acc", "HOLDOUT_4_INPUT_acc",
        "MULTI_3_INPUT_acc", "HOLDOUT_3_INPUT_acc",
        "MULTI_6_INPUT_acc",
        "UNSEEN_4_acc", "UNSEEN_6_acc"
        ])
        writer.writerow([
        overall_acc,
        op_acc.get("MULTI_4_INPUT", 0.0),
        op_acc.get("HOLDOUT_4_INPUT", 0.0),
        op_acc.get("MULTI_3_INPUT", 0.0),
        op_acc.get("HOLDOUT_3_INPUT", 0.0),
        op_acc.get("MULTI_6_INPUT", 0.0),
        op_acc.get("UNSEEN_4", 0.0),
        op_acc.get("UNSEEN_6", 0.0),
    ])

    print("generate samples from test prompts:")
    for _ in range(15):
        prompt, true_answer, operation = random.choice(test_rows)
        prompt_tokens = tokenize_bool(prompt)
        idx = torch.tensor([encode_tokens(prompt_tokens)], dtype=torch.long, device=device)
        output_idx = model.generate_greedy(idx, max_new_tokens=1)
        out_pred = decode_tokens(output_idx[0].tolist())
        pred_answer = out_pred[len(prompt_tokens)]
        
        equation_full = prompt + pred_answer
        print(f"[{operation}] {equation_full} (pred: {pred_answer}, true: {true_answer})")