import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F

names = [
    "Alexander", "Sophia", "Liam", "Olivia", "Ethan",
    "Emma", "Noah", "Ava", "Mason", "Isabella",
    "Logan", "Mia", "Lucas", "Harper", "Elijah",
    "Evelyn", "Aiden", "Abigail", "Caleb", "Emily",
    "Benjamin", "Elizabeth", "Jackson", "Mila", "Sebastian",
    "Ella", "Mateo", "Avery", "Jack", "Sofia",
    "Owen", "Camila", "Theodore", "Aria", "Julian",
    "Scarlett", "Asher", "Victoria", "Leo", "Madison",
    "Gabriel", "Luna", "Ezra", "Chloe", "Hudson",
    "Penelope", "Christian", "Layla", "Lincoln", "Nora"
]

names = ["..." + name.lower() + "." for name in names]

X = []
Y = []

for name in names:
    for i in range(len(name) - 3):
        X.append(name[i:i + 3])
        Y.append(name[i + 3])

letter_list = ".abcdefghijklmnoprstquvwxyz"
letterstoi = {letter: i for i, letter in enumerate(letter_list)}

X_index = [[letterstoi[letter] for letter in x] for x in X]
Y_index = [letterstoi[y] for y in Y]

X_tensor = torch.tensor(X_index)
Y_tensor = torch.tensor(Y_index)

n = X_tensor.shape[0]
perm = torch.randperm(n)

n1 = int(0.8 * n)
n2 = int(0.9 * n)

train_idx = perm[:n1]
dev_idx = perm[n1:n2]
test_idx = perm[n2:]

X_train = X_tensor[train_idx]
Y_train = Y_tensor[train_idx]

X_dev = X_tensor[dev_idx]
Y_dev = Y_tensor[dev_idx]

X_test = X_tensor[test_idx]
Y_test = Y_tensor[test_idx]

print("train:", X_train.shape, Y_train.shape)
print("dev:", X_dev.shape, Y_dev.shape)
print("test:", X_test.shape, Y_test.shape)

torch.manual_seed(42)

batch_size = 32
vocab_size = 27
embed_dim = 10
hidden_dim = 200
block_size = 3


ix = torch.randint(0, X_train.shape[0], (batch_size,))

X_batch = X_train[ix]
Y_batch = Y_train[ix]

embed_table = torch.randn((vocab_size, embed_dim), requires_grad=True)

W1 = (torch.randn((block_size * embed_dim, hidden_dim)) * (5 / 3) / ((block_size * embed_dim) ** 0.5)).requires_grad_()
b1 = (torch.randn(hidden_dim) * 0.01).requires_grad_()

bn_gain = torch.ones(hidden_dim, requires_grad=True)
bn_bias = torch.zeros(hidden_dim, requires_grad=True)

W2 = (torch.randn((hidden_dim, vocab_size)) * 0.01).requires_grad_()
b2 = torch.zeros(vocab_size, requires_grad=True)

embedding = embed_table[X_batch]
hpre = embedding.flatten(1, 2) @ W1 + b1

bn_mean = hpre.mean(0, keepdim=True)
bn_var = hpre.var(0, keepdim=True, unbiased=False)
bn_std = (bn_var + 1e-5).sqrt()
bn_norm = (hpre - bn_mean) / bn_std

h = bn_gain * bn_norm + bn_bias
h_tanh = torch.tanh(h)

logits = h_tanh @ W2 + b2

counts = logits.exp()
counts_sum = counts.sum(1, keepdim=True)

probs = counts / counts_sum

logprobs = probs.log()
loss = -logprobs[torch.arange(batch_size), Y_batch].mean()

logits.retain_grad()
counts.retain_grad()
counts_sum.retain_grad()
probs.retain_grad()
logprobs.retain_grad()

h_tanh.retain_grad()
h.retain_grad()
bn_norm.retain_grad()
bn_std.retain_grad()
bn_var.retain_grad()
bn_mean.retain_grad()
hpre.retain_grad()
embedding.retain_grad()

for p in [embed_table, W1, b1, bn_gain, bn_bias, W2, b2]:
    p.grad = None

loss.backward()

print("loss:", loss.item())

print("\nlogprobs.grad")
print(logprobs.grad)

print("\nprobs.grad")
print(probs.grad)

print("\ncounts.grad")
print(counts.grad)

print("\ncounts_sum.grad")
print(counts_sum.grad)

print("\nlogits.grad")
print(logits.grad)

print("\nh_tanh.grad")
print(h_tanh.grad)

print("\nh.grad")
print(h.grad)

print("\nbn_norm.grad")
print(bn_norm.grad)

print("\nbn_std.grad")
print(bn_std.grad)

print("\nbn_var.grad")
print(bn_var.grad)

print("\nbn_mean.grad")
print(bn_mean.grad)

print("\nhpre.grad")
print(hpre.grad)

print("\nembedding.grad")
print(embedding.grad)

print("\nW2.grad")
print(W2.grad)

print("\nb2.grad")
print(b2.grad)

print("\nbn_gain.grad")
print(bn_gain.grad)

print("\nbn_bias.grad")
print(bn_bias.grad)

print("\nW1.grad")
print(W1.grad)

print("\nb1.grad")
print(b1.grad)

print("\nembed_table.grad")
print(embed_table.grad)

def cmp(s, dt, t):
    ex = torch.allclose(dt, t, atol=1e-5, rtol=1e-5)
    maxdiff = (dt - t).abs().max().item()
    mean_diff = (dt - t).abs().mean().item()
    print(f"{s:20s} | exact: {ex} | max diff: {maxdiff:.10f} | mean diff: {mean_diff:.10f}")

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[torch.arange(batch_size), Y_batch] = -1 / batch_size
cmp("logprobs", dlogprobs, logprobs.grad)

dprobs = 1 / probs * dlogprobs
cmp("probs", dprobs, probs.grad)

dcounts = (1 / counts_sum) * dprobs
cmp("counts", dcounts, counts.grad)

dcounts_sum = (-counts / counts_sum**2 * dprobs).sum(1, keepdim=True)
cmp("counts_sum", dcounts_sum, counts_sum.grad)

dcounts = dcounts + torch.ones_like(counts) * dcounts_sum
cmp("counts", dcounts, counts.grad)

dlogits = counts * dcounts
cmp("logits", dlogits, logits.grad)

dh_tanh = dlogits @ W2.T
cmp("h_tanh", dh_tanh, h_tanh.grad)

dh = (1 - h_tanh**2) * dh_tanh
cmp("h", dh, h.grad)

dbn_gain = (bn_norm * dh).sum(0, keepdim=True)
cmp("bn_gain", dbn_gain, bn_gain.grad)

dbn_norm = bn_gain * dh
cmp("bn_norm", dbn_norm, bn_norm.grad)

dbn_std = ((hpre - bn_mean) / bn_std**2 * -1 * dbn_norm).sum(0, keepdim=True)
cmp("bn_std", dbn_std, bn_std.grad)

dbn_var = dbn_std * 0.5 / bn_std
cmp("bn_var", dbn_var, bn_var.grad)

dbn_mean = (-dbn_norm / bn_std).sum(0, keepdim=True)
cmp("bn_mean", dbn_mean, bn_mean.grad)

N = hpre.shape[0]
dhpre = dbn_norm / bn_std
dhpre += dbn_var * 2 * (hpre - bn_mean) / N
dhpre += dbn_mean / N
cmp("hpre", dhpre, hpre.grad)

dembedding_flat = dhpre @ W1.T
dembedding = dembedding_flat.view(embedding.shape)

cmp("embedding", dembedding, embedding.grad)

dW2 = h_tanh.T @ dlogits
cmp("W2", dW2, W2.grad)

db2 = dlogits.sum(0)
cmp("b2", db2, b2.grad)

dW1 = embedding.flatten(1, 2).T @ dhpre
cmp("W1", dW1, W1.grad)

db1 = dhpre.sum(0)
cmp("b1", db1, b1.grad)