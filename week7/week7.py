from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import urllib.request

torch.manual_seed(1337)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = urllib.request.urlopen(url).read().decode("utf-8")

print(len(text))
print(text[:500])

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(vocab_size)
print(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape)
print(data[:100])

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

print(len(train_data))
print(len(val_data))

batch_size = 32
block_size = 8

def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y

xb, yb = get_batch("train")

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

model = BigramLanguageModel(vocab_size)
logits, loss = model(xb, yb)

@torch.no_grad()
def generate(idx, max_new_tokens):

    for _ in range(max_new_tokens):

        logits, loss = model(idx)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

context = torch.zeros((1, 1), dtype=torch.long)

print("Before")
print(decode(generate(context, 200)[0].tolist()))

eval_iters = 200


@torch.no_grad()
def estimate_loss():

    out = {}
    model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()
    return out

optimizer = torch.optim.AdamW( model.parameters(), lr=1e-2)

max_iters = 5000
eval_interval = 500

train_losses = []
val_losses = []
steps = []


for iter in range(max_iters):

    if iter % eval_interval == 0:

        losses = estimate_loss()

        train_loss = losses["train"].item()
        val_loss = losses["val"].item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        steps.append(iter)

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()

plt.plot(steps, train_losses, label="train")

plt.plot(steps, val_losses, label="validation")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()

plt.show()

context = torch.zeros((1, 1), dtype=torch.long)

generated = generate(context, max_new_tokens=500)

print("After")

print(decode(generated[0].tolist()))

B = 4
T = 8
C = 2

x = torch.randn(B, T, C)
print(x.shape)

xbow = torch.zeros((B, T, C))

for b in range(B):
    for t in range(T):

        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, dim=0)

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)

xbow2 = wei @ x

print("Matrix",torch.allclose(xbow, xbow2))

wei = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(wei == 0,float("-inf"))
wei = F.softmax(wei, dim=1)

xbow3 = wei @ x

print("For Softmax",torch.allclose(xbow, xbow3))
print("Matrix Softmax",torch.allclose(xbow2, xbow3))

head_size = 16

token_embedding = nn.Embedding(vocab_size, C)
position_embedding = nn.Embedding(T, C)

idx = torch.randint(vocab_size, (B, T))
tok_emb = token_embedding(idx)

pos = torch.arange(T)
pos_emb = position_embedding(pos)

x = tok_emb + pos_emb

print("x shape", x.shape)

query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

q = query(x)
k = key(x)
v = value(x)

print("q ", q.shape)
print("k ", k.shape)
print("v ", v.shape)

wei = q @ k.transpose(-2, -1)
print("wei shape ", wei.shape)

wei = wei * head_size**-0.5
tril = torch.tril(torch.ones(T, T))

wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

print("wei")
print(wei[0])

out = wei @ v
print("out shape ", out.shape)

scores = torch.tensor([1.0, 2.0, 3.0, 4.0])

softmax_without_scaling = F.softmax(
    scores,
    dim=-1
)

softmax_with_scaling = F.softmax(
    scores / (head_size ** 0.5),
    dim=-1
)

print("Scores ")
print(scores)

print("Softmax no scaling ")
print(softmax_without_scaling)

print("Softmax sqrt(head_size) scaling")
print(softmax_with_scaling)

class Head(nn.Module):

    def __init__(self, head_size):

        super().__init__()

        self.key = nn.Linear(C, head_size, bias=False)
        self.query = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(T, T)))

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * C**-0.5

        wei = wei.masked_fill( self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out

class BigramAttentionLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            C
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            C
        )

        self.sa_head = Head(head_size)

        self.lm_head = nn.Linear(
            head_size,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb
        x = self.sa_head(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None

        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

attention_model = BigramAttentionLanguageModel(vocab_size)
print(attention_model)

xb, yb = get_batch("train")
logits, loss = attention_model(xb, yb)

print("Initial loss", loss.item())

optimizer = torch.optim.AdamW(
    attention_model.parameters(),
    lr=1e-3
)

max_iters = 5000
eval_interval = 500

attention_train_losses = []
attention_val_losses = []
attention_steps = []


@torch.no_grad()
def estimate_attention_loss():

    out = {}

    attention_model.eval()

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            X, Y = get_batch(split)

            logits, loss = attention_model(X, Y)

            losses[k] = loss.item()

        out[split] = losses.mean()

    attention_model.train()

    return out


for iter in range(max_iters):

    if iter % eval_interval == 0:

        losses = estimate_attention_loss()

        train_loss = losses["train"].item()
        val_loss = losses["val"].item()

        attention_train_losses.append(train_loss)
        attention_val_losses.append(val_loss)
        attention_steps.append(iter)


    xb, yb = get_batch("train")
    logits, loss = attention_model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

plt.plot(steps, val_losses,label="Bigram")
plt.plot(attention_steps, attention_val_losses, label="Bigram + Self-Attention")

plt.xlabel("Step")
plt.ylabel("Validation Loss")

plt.legend()

plt.show()

@torch.no_grad()
def generate_attention(idx, max_new_tokens):

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -block_size:]
        logits, loss = attention_model(idx_cond)
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

context = torch.zeros((1, 1),dtype=torch.long)
generated = generate_attention(context, max_new_tokens=500)

print(decode(generated[0].tolist()))

