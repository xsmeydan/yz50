
import math
import time
import random
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt


SEED = 1337
BATCH_SIZE = 32
BLOCK_SIZE = 64
STAGE_ITERS = 1200
SCALED_ITERS = 2000
TURKISH_ITERS = 2000
EVAL_ITERS = 50
EVAL_INTERVAL = 300

N_EMBD = 32
N_HEAD = 4
N_LAYER = 1
DROPOUT = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def load_shakespeare():
    url = (
        "https://raw.githubusercontent.com/karpathy/"
        "char-rnn/master/data/tinyshakespeare/input.txt"
    )
    path = Path("input.txt")

    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = urllib.request.urlopen(url).read().decode("utf-8")
        path.write_text(text, encoding="utf-8")

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    def encode(s):
        return [stoi[ch] for ch in s if ch in stoi]

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))

    return (
        text,
        len(chars),
        encode,
        decode,
        data[:split],
        data[split:],
        chars,
    )


TEXT, VOCAB_SIZE, encode, decode, train_data, val_data, chars = (
    load_shakespeare()
)


def make_batch_fn(train_tokens, val_tokens, block_size, batch_size):
    def get_batch(split):
        source = train_tokens if split == "train" else val_tokens

        if len(source) <= block_size + 1:
            raise ValueError("Veri seti block_size için çok küçük.")

        ix = torch.randint(
            len(source) - block_size - 1,
            (batch_size,)
        )
        x = torch.stack(
            [source[i:i + block_size] for i in ix]
        )
        y = torch.stack(
            [source[i + 1:i + block_size + 1] for i in ix]
        )

        return x.to(DEVICE), y.to(DEVICE)

    return get_batch


get_batch = make_batch_fn(
    train_data, val_data, BLOCK_SIZE, BATCH_SIZE
)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.block_size = BLOCK_SIZE
        self.token_embedding_table = nn.Embedding(
            vocab_size, vocab_size
        )

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss


class Head(nn.Module):
    def __init__(
        self, n_embd, head_size, block_size, dropout=0.0
    ):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self, n_embd, n_head, block_size, dropout=0.0
    ):
        super().__init__()
        assert n_embd % n_head == 0

        head_size = n_embd // n_head

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, n_embd, n_head, block_size, dropout=0.1
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(
            n_embd, n_head, block_size, dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class StageLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=32,
        n_head=4,
        n_layer=1,
        block_size=64,
        dropout=0.0,
        stage="layernorm"
    ):
        super().__init__()
        self.block_size = block_size
        self.stage = stage

        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )

        if stage == "single":
            self.attention = Head(
                n_embd, n_embd, block_size, dropout
            )
        elif stage != "layernorm":
            self.attention = MultiHeadAttention(
                n_embd, n_head, block_size, dropout
            )

        if stage == "ffn":
            self.ffwd = FeedForward(n_embd, dropout)

        if stage == "residual":
            self.ffwd = FeedForward(n_embd, dropout)

        if stage == "layernorm":
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    n_embd, n_head, block_size, dropout
                )
                for _ in range(n_layer)
            ])
            self.final_ln = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape

        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )
        x = tok + pos

        if self.stage == "single":
            x = self.attention(x)

        elif self.stage == "multihead":
            x = self.attention(x)

        elif self.stage == "ffn":
            x = self.ffwd(self.attention(x))

        elif self.stage == "residual":
            x = x + self.attention(x)
            x = x + self.ffwd(x)

        elif self.stage == "layernorm":
            for block in self.blocks:
                x = block(x)
            x = self.final_ln(x)

        logits = self.lm_head(x)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss


@torch.no_grad()
def estimate_loss(model, get_batch_fn, eval_iters=EVAL_ITERS):
    model.eval()
    result = {}

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            xb, yb = get_batch_fn(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()

        result[split] = losses.mean().item()

    model.train()
    return result


def train_model(
    model,
    get_batch_fn,
    max_iters,
    eval_interval=EVAL_INTERVAL,
    learning_rate=1e-3
):
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )

    history = []
    start = time.time()

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, get_batch_fn)
            history.append({
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"]
            })
            print(
                f"{step:5d} | "
                f"train={losses['train']:.4f} | "
                f"val={losses['val']:.4f}"
            )

        xb, yb = get_batch_fn("train")
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start
    final = estimate_loss(model, get_batch_fn)

    return model, pd.DataFrame(history), final, elapsed


@torch.no_grad()
def generate_text(
    model, decode_fn, max_new_tokens=500, start_id=0
):
    model.eval()
    idx = torch.tensor(
        [[start_id]], dtype=torch.long, device=DEVICE
    )

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)

    model.train()
    return decode_fn(idx[0].tolist())


results = []
models = {}
histories = {}


torch.manual_seed(SEED)
bigram = BigramLanguageModel(VOCAB_SIZE).to(DEVICE)

bigram, hist, final, elapsed = train_model(
    bigram, get_batch, STAGE_ITERS
)

models["Bigram"] = bigram
histories["Bigram"] = hist
results.append({
    "Model": "Bigram",
    "Parameters": sum(p.numel() for p in bigram.parameters()),
    "Train Loss": final["train"],
    "Validation Loss": final["val"],
    "Training Time (s)": round(elapsed, 2)
})


stages = [
    ("Single-head", "single"),
    ("Multi-head", "multihead"),
    ("+ FeedForward", "ffn"),
    ("+ Residual", "residual"),
    ("+ LayerNorm", "layernorm")
]

for label, stage in stages:
    print("\nTraining:", label)
    torch.manual_seed(SEED)

    model = StageLanguageModel(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT,
        stage=stage
    )

    model, hist, final, elapsed = train_model(
        model, get_batch, STAGE_ITERS
    )

    models[label] = model
    histories[label] = hist

    results.append({
        "Model": label,
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Train Loss": final["train"],
        "Validation Loss": final["val"],
        "Training Time (s)": round(elapsed, 2)
    })

comparison_df = pd.DataFrame(results)
print("\nArchitecture comparison")
print(comparison_df.round(4).to_string(index=False))
comparison_df.to_csv(
    "week8_architecture_comparison.csv", index=False
)

plt.figure(figsize=(10, 5))
for label, hist in histories.items():
    plt.plot(
        hist["step"], hist["val_loss"],
        marker="o", label=label
    )
plt.xlabel("Training step")
plt.ylabel("Validation loss")
plt.title("Architecture vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("week8_loss_comparison.png")
plt.show()


scaled_config = {
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 2,
    "block_size": BLOCK_SIZE,
    "dropout": 0.1
}

torch.manual_seed(SEED)
scaled_model = StageLanguageModel(
    vocab_size=VOCAB_SIZE,
    stage="layernorm",
    **scaled_config
)

scaled_model, scaled_history, scaled_final, scaled_time = train_model(
    scaled_model,
    get_batch,
    max_iters=SCALED_ITERS,
    eval_interval=400,
    learning_rate=3e-4
)

scaled_params = sum(
    p.numel() for p in scaled_model.parameters()
)

scale_summary = pd.DataFrame([{
    **scaled_config,
    "Parameters": scaled_params,
    "Train Loss": scaled_final["train"],
    "Validation Loss": scaled_final["val"],
    "Training Time (s)": round(scaled_time, 2)
}])

print("\nScaled model results")
print(scale_summary.round(4).to_string(index=False))
scale_summary.to_csv("week8_scaled_model.csv", index=False)

print("\nGenerated Shakespeare text")
print(generate_text(scaled_model, decode, 700))

scaled_history.to_csv("week8_scaled_history.csv", index=False)


x_demo = torch.randn(4, 8, 32)
layer_norm = nn.LayerNorm(32)
batch_norm = nn.BatchNorm1d(32)

ln_out = layer_norm(x_demo)
bn_out = batch_norm(x_demo.transpose(1, 2)).transpose(1, 2)

print("\nNormalization check")
print("Input shape:", x_demo.shape)
print("LayerNorm shape:", ln_out.shape)
print("BatchNorm shape:", bn_out.shape)
print("LayerNorm running_mean:", hasattr(layer_norm, "running_mean"))
print("BatchNorm running_mean:", hasattr(batch_norm, "running_mean"))


parallel_attention = None


class ParallelMultiHeadAttention(nn.Module):
    def __init__(
        self, n_embd, n_head, block_size, dropout=0.0
    ):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_size = n_embd // n_head

        self.c_attn = nn.Linear(
            n_embd, 3 * n_embd, bias=False
        )
        self.proj = nn.Linear(n_embd, n_embd)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)

        q = q.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)
        k = k.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)
        v = v.view(
            B, T, self.n_head, self.head_size
        ).transpose(1, 2)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (self.head_size ** -0.5)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


reference = models["Multi-head"].attention
parallel_attention = ParallelMultiHeadAttention(
    N_EMBD, N_HEAD, BLOCK_SIZE, dropout=0.0
).to(DEVICE)

with torch.no_grad():
    q_weights = torch.cat(
        [head.query.weight for head in reference.heads], dim=0
    )
    k_weights = torch.cat(
        [head.key.weight for head in reference.heads], dim=0
    )
    v_weights = torch.cat(
        [head.value.weight for head in reference.heads], dim=0
    )

    parallel_attention.c_attn.weight.copy_(
        torch.cat([q_weights, k_weights, v_weights], dim=0)
    )
    parallel_attention.proj.load_state_dict(
        reference.proj.state_dict()
    )

reference.eval()
parallel_attention.eval()

test_x = torch.randn(2, 10, N_EMBD, device=DEVICE)

with torch.no_grad():
    expected = reference(test_x)
    actual = parallel_attention(test_x)

max_difference = (expected - actual).abs().max().item()
print("\nParallel attention check")
print("Maximum absolute difference:", max_difference)
print(
    "Equivalent within tolerance:",
    torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)
)


turkish_path = Path("turkish.txt")

if turkish_path.exists():
    turkish_text = turkish_path.read_text(encoding="utf-8")
    tr_chars = sorted(set(turkish_text))
    tr_vocab_size = len(tr_chars)
    tr_stoi = {ch: i for i, ch in enumerate(tr_chars)}
    tr_itos = {i: ch for ch, i in tr_stoi.items()}

    def tr_encode(s):
        return [tr_stoi[ch] for ch in s if ch in tr_stoi]

    def tr_decode(ids):
        return "".join(tr_itos[int(i)] for i in ids)

    tr_data = torch.tensor(tr_encode(turkish_text), dtype=torch.long)
    tr_split = int(0.9 * len(tr_data))
    tr_train_data = tr_data[:tr_split]
    tr_val_data = tr_data[tr_split:]

    tr_get_batch = make_batch_fn(
        tr_train_data, tr_val_data, BLOCK_SIZE, BATCH_SIZE
    )

    torch.manual_seed(SEED)
    turkish_model = StageLanguageModel(
        vocab_size=tr_vocab_size,
        n_embd=64,
        n_head=4,
        n_layer=2,
        block_size=BLOCK_SIZE,
        dropout=0.1,
        stage="layernorm"
    )

    turkish_model, turkish_history, turkish_final, turkish_time = (
        train_model(
            turkish_model,
            tr_get_batch,
            max_iters=TURKISH_ITERS,
            eval_interval=400,
            learning_rate=3e-4
        )
    )

    turkish_params = sum(
        p.numel() for p in turkish_model.parameters()
    )

    turkish_summary = pd.DataFrame([{
        "Dataset": "Turkish",
        "Unique characters": tr_vocab_size,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "block_size": BLOCK_SIZE,
        "Parameters": turkish_params,
        "Train Loss": turkish_final["train"],
        "Validation Loss": turkish_final["val"],
        "Training Time (s)": round(turkish_time, 2)
    }])

    print("\nTurkish model results")
    print(turkish_summary.round(4).to_string(index=False))
    turkish_summary.to_csv("week8_turkish_model.csv", index=False)
    turkish_history.to_csv("week8_turkish_history.csv", index=False)

    print("\nGenerated Turkish text")
    print(generate_text(
        turkish_model,
        tr_decode,
        700,
        tr_stoi.get(" ", 0)
    ))
else:
    print(
        "\nTurkish dataset skipped: place turkish.txt "
        "in the current working directory to train it."
    )


print("\nSaved output files:")
for path in sorted(Path(".").glob("week8_*")):
    print(path.name)
