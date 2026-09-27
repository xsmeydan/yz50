from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F


class Module:
    def __init__(self):
        self.training = True

    def train(self):
        self.training = True

        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.train()
            elif isinstance(value, list):
                for layer in value:
                    if isinstance(layer, Module):
                        layer.train()

    def eval(self):
        self.training = False

        for value in self.__dict__.values():
            if isinstance(value, Module):
                value.eval()
            elif isinstance(value, list):
                for layer in value:
                    if isinstance(layer, Module):
                        layer.eval()


class Linear(Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()

        self.W = (
            torch.randn(fan_in, fan_out)
            * (5 / 3)
            / (fan_in ** 0.5)
        ).requires_grad_()

        self.b = torch.zeros(fan_out).requires_grad_()

    def __call__(self, x):
        self.out = x @ self.W + self.b
        return self.out

    def parameters(self):
        return [self.W, self.b]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.gain = torch.ones(dim, requires_grad=True)
        self.bias = torch.zeros(dim, requires_grad=True)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            mean = x.mean(0, keepdim=True)
            var = x.var(0, keepdim=True)

            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean
                )

                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var
                )
        else:
            mean = self.running_mean
            var = self.running_var

        xhat = (x - mean) / torch.sqrt(var + self.eps)

        self.out = self.gain * xhat + self.bias

        return self.out

    def parameters(self):
        return [self.gain, self.bias]


class Tanh(Module):
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding(Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()

        self.weight = torch.randn(
            vocab_size,
            embed_dim,
            requires_grad=True
        )

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten(Module):
    def __call__(self, x):
        self.out = x.view(x.size(0), -1)
        return self.out

    def parameters(self):
        return []


class Sequential(Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        parameters = []

        for layer in self.layers:
            parameters.extend(layer.parameters())

        return parameters


# DATASET

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

names = [name.lower() + "." for name in names]

chars = sorted(set(''.join(names)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

vocab_size = len(chars)

print("vocab_size:", vocab_size)




# BUILD DATASET 3

block_size = 3


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w:
            ix = stoi[ch]

            X.append(context)
            Y.append(ix)

            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)


n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])

print("Xtr:", Xtr.shape)
print("Ytr:", Ytr.shape)

# MODEL

model = Sequential([
    Embedding(vocab_size, 2),
    Flatten(),
    Linear(6, 100),
    BatchNorm1d(100),
    Tanh(),
    Linear(100, vocab_size)
])

# PARAMETERS

parameters = model.parameters()

print("number of parameters:", sum(p.nelement() for p in parameters))

# TRAINING

max_steps = 200000
batch_size = 32

lossi = []

model.train()

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)

    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None

    loss.backward()
    lr = 0.1 if i < 150000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())

    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")

# LOSS CURVE

plt.plot(
    torch.tensor(lossi)
    .view(-1, 1000)
    .mean(1)
)

plt.show()

# DEV LOSS

model.eval()

with torch.no_grad():
    logits = model(Xdev)
    dev_loss = F.cross_entropy(logits, Ydev)

print("dev loss:", dev_loss.item())




# BUILD DATASET 8

block_size = 8


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w:
            ix = stoi[ch]

            X.append(context)
            Y.append(ix)

            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)


n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])

print("Xtr:", Xtr.shape)
print("Ytr:", Ytr.shape)

# MODEL

model = Sequential([
    Embedding(vocab_size, 2),
    Flatten(),
    Linear(16, 100),
    BatchNorm1d(100),
    Tanh(),
    Linear(100, vocab_size)
])

# PARAMETERS

parameters = model.parameters()

print("number of parameters:", sum(p.nelement() for p in parameters))

# TRAINING

max_steps = 200000
batch_size = 32

lossi = []

model.train()

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)

    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None

    loss.backward()
    lr = 0.1 if i < 150000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())

    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")

# LOSS CURVE

plt.plot(
    torch.tensor(lossi)
    .view(-1, 1000)
    .mean(1)
)

plt.show()

# DEV LOSS

model.eval()

with torch.no_grad():
    logits = model(Xdev)
    dev_loss = F.cross_entropy(logits, Ydev)

print("dev loss:", dev_loss.item())

# Wavenet
class FlattenConsecutive(Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)

        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out

    def parameters(self):
        return []


block_size = 8


def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size

        for ch in w:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    return torch.tensor(X), torch.tensor(Y)


n1 = int(0.8 * len(names))
n2 = int(0.9 * len(names))

Xtr, Ytr = build_dataset(names[:n1])
Xdev, Ydev = build_dataset(names[n1:n2])
Xte, Yte = build_dataset(names[n2:])


model = Sequential([
    Embedding(vocab_size, 2),
    FlattenConsecutive(2),
    Linear(4, 100),
    BatchNorm1d(100),
    Tanh(),
    FlattenConsecutive(2),
    Linear(200, 100),
    BatchNorm1d(100),
    Tanh(),
    FlattenConsecutive(2),
    Linear(200, 100),
    BatchNorm1d(100),
    Tanh(),
    Linear(100, vocab_size)
])


x = Xtr[:32]

for layer in model.layers:
    x = layer(x)
    print(type(layer).__name__, x.shape)


parameters = model.parameters()

print("number of parameters:", sum(p.nelement() for p in parameters))


max_steps = 200000
batch_size = 32

lossi = []

model.train()

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 150000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())

    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")


plt.plot(
    torch.tensor(lossi).view(-1, 1000).mean(1)
)

plt.show()


model.eval()

with torch.no_grad():
    logits = model(Xdev)
    dev_loss_before = F.cross_entropy(logits, Ydev)

print("dev loss before BN fix:", dev_loss_before.item())


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.gain = torch.ones(dim, requires_grad=True)
        self.bias = torch.zeros(dim, requires_grad=True)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if x.ndim == 2:
            dim = 0
        elif x.ndim == 3:
            dim = (0, 1)
        else:
            raise ValueError(f"Unsupported input dimension: {x.ndim}")

        if self.training:
            mean = x.mean(dim, keepdim=True)
            var = x.var(dim, keepdim=True, unbiased=False)

            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean
                    + self.momentum * mean.squeeze()
                )

                self.running_var = (
                    (1 - self.momentum) * self.running_var
                    + self.momentum * var.squeeze()
                )
        else:
            if x.ndim == 2:
                mean = self.running_mean
                var = self.running_var
            else:
                mean = self.running_mean.view(1, 1, -1)
                var = self.running_var.view(1, 1, -1)

        xhat = (x - mean) / torch.sqrt(var + self.eps)

        self.out = self.gain * xhat + self.bias

        return self.out

    def parameters(self):
        return [self.gain, self.bias]


model = Sequential([
    Embedding(vocab_size, 2),
    FlattenConsecutive(2),
    Linear(4, 100),
    BatchNorm1d(100),
    Tanh(),
    FlattenConsecutive(2),
    Linear(200, 100),
    BatchNorm1d(100),
    Tanh(),
    FlattenConsecutive(2),
    Linear(200, 100),
    BatchNorm1d(100),
    Tanh(),
    Linear(100, vocab_size)
])


parameters = model.parameters()

print(
    "number of parameters after BN fix:",
    sum(p.nelement() for p in parameters)
)


max_steps = 200000
batch_size = 32

lossi = []

model.train()

for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if i < 150000 else 0.01

    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())

    if i % 10000 == 0:
        print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")


plt.plot(
    torch.tensor(lossi).view(-1, 1000).mean(1)
)

plt.show()


model.eval()

with torch.no_grad():
    logits = model(Xdev)
    dev_loss_after = F.cross_entropy(logits, Ydev)

print("dev loss after BN fix:", dev_loss_after.item())

print("before:", dev_loss_before.item())
print("after:", dev_loss_after.item())