import math
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
import csv

#final_assignment\MathGPT\add_sub_div_mul\add_sub_div_mul_dataset\maths_add_sub_div_mul_train.txt

#script toggles for training and testing
#TRAIN False if only testing a pre-trained model
#TEST False if only training without testing
#BOTH TRUE to do both training and testing at the same time
TRAIN=True
TEST=True

# --- paths & dirs ---------------------------------------------------------

# folder that this MathGPT.py file is in
BASE_DIR = os.path.dirname(__file__)

# where to save models (same folder as script, or change if you want)
MODEL_DIR = BASE_DIR
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_BASENAME   = "mathGPT_add_sub_div_mul"
MODEL_PATH       = os.path.join(MODEL_DIR, MODEL_BASENAME)
MODEL_BEST_PATH  = os.path.join(MODEL_DIR, MODEL_BASENAME + "_best.pt")
MODEL_LAST_PATH  = os.path.join(MODEL_DIR, MODEL_BASENAME + "_last.pt")

# dataset lives in subfolder add_sub_div_mul_dataset/
DATASET_DIR = os.path.join(BASE_DIR, "add_sub_div_mul_dataset")
TRAIN_PATH  = os.path.join(DATASET_DIR, "maths_add_sub_div_mul_train.txt")
TEST_PATH   = os.path.join(DATASET_DIR, "maths_add_sub_div_mul_test.tsv")

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 8
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#point to the mathGPT train set
with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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
            nn.ReLU(), #changed from RELu to GeLU - better performance in transformers apparently
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
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)
        
        #weight tying implementation
        self.lm_head.weight = self.token_embedding_table.weight

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
    
    def generate_greedy(self,idx,max_new_tokens):
        """version of generate that uses greedy sampling - always picks the most probable next token - /n
        this is more applicable to maths problems where there is a single correct answer"""
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            # sample from the distribution
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

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
        xb, yb = get_batch('train')

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

@torch.no_grad()
def test_model(model,test_rows,max_new_tokens=4):
    model.eval()
    total = 0
    correct = 0
    operation_total = {"+":0, "-":0,"*":0,"/":0,"INT":0,"DER":0}
    operation_correct = {"+":0,"-":0,"*":0,"/":0,"INT":0,"DER":0}
    
    for prompt, answer, operation in test_rows:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        
        output_idx = model.generate_greedy(idx, max_new_tokens=max_new_tokens)
        full_output = decode(output_idx[0].tolist())
        
        pred_full_suffix = full_output[len(prompt):]
        
        #extract pred answer
        pred_answer = pred_full_suffix[:len(answer)]
        
        is_correct = (pred_answer == answer)
        total += 1
        correct += int(is_correct)
        
        if operation in operation_total:
            operation_total[operation] += 1
            operation_correct[operation] += int(is_correct)
            
    #overall_acc = correct / total * 100
    #print(f"Overall Test Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    
    overall_acc = correct / total if total > 0 else 0.0
    add_acc = (
        operation_correct["+"] / operation_total["+"]
        if operation_total["+"] > 0 else 0.0
    )
    sub_acc = (
        operation_correct["-"] / operation_total["-"]
        if operation_total["-"] > 0 else 0.0
    )
    mul_acc = (
        operation_correct["*"] / operation_total["*"]
        if operation_total["*"] > 0 else 0.0
    )
    div_acc = (
        operation_correct["/"] / operation_total["/"]
        if operation_total["/"] > 0 else 0.0
    )
    int_acc = (
        operation_correct["INT"] / operation_total["INT"]
        if operation_total["INT"] > 0 else 0.0
    )
    der_acc = (
        operation_correct["DER"] / operation_total["DER"]
        if operation_total["DER"] > 0 else 0.0
    )
    
        
    print("\n=== MathGPT test results ===")
    print(f"Overall equation accuracy: {overall_acc:.4f}")
    print(f"Accuracy for '+':          {add_acc:.4f}")
    print(f"Accuracy for '-':          {sub_acc:.4f}\n")
    print(f"Accuracy for '*':          {mul_acc:.4f}")
    print(f"Accuracy for '/':          {div_acc:.4f}")
    print(f"Accuracy for 'INT':        {int_acc:.4f}")
    print(f"Accuracy for 'DER':        {der_acc:.4f}\n")

    model.train()
    return overall_acc, {"+": add_acc, "-": sub_acc, "*": mul_acc, "/": div_acc, "INT": int_acc, "DER": der_acc}   

if TEST:
    # always evaluate best checkpoint
    state_dict = torch.load(MODEL_BEST_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print("loaded model from", MODEL_BEST_PATH)
    
    test_rows = load_testset(TEST_PATH)
    print(f"Loaded {len(test_rows)} test examples from {TEST_PATH}")
    
    overall_acc, op_acc = test_model(model, test_rows, max_new_tokens=4)

    with open(os.path.join(MODEL_DIR, MODEL_BASENAME + "_test_results.csv"),
              mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["overall_acc", "add_acc", "sub_acc", "mul_acc", "div_acc", "int_acc", "der_acc"])
        writer.writerow([overall_acc, op_acc.get("+", 0.0), op_acc.get("-", 0.0), op_acc.get("*", 0.0), op_acc.get("/", 0.0), op_acc.get("INT", 0.0), op_acc.get("DER", 0.0)])

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("Sample generation:")
    print(decode(model.generate_greedy(context, max_new_tokens=200)[0].tolist()))