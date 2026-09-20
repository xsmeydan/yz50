import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


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


for i in range(len(names)):
    names[i] = "..." + names[i].lower() + "."


X = []
Y = []

for name in names:
    for i in range(len(name) - 3):
        trigram = name[i:i + 3]
        X.append(trigram)
        Y.append(name[i + 3])


letter_list = ".abcdefghijklmnoprstquvwxyz"
letterstoi = {letter: i for i, letter in enumerate(letter_list)}


X_index = []

for x in X:
    temp = []
    for letter in x:
        temp.append(letterstoi[letter])
    X_index.append(temp)


Y_index = []

for y in Y:
    Y_index.append(letterstoi[y])


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


embed_dim = 2
hidden_dim = 100

embed_table = torch.randn((27, embed_dim), requires_grad=True)
W1 = torch.randn((3 * embed_dim, hidden_dim), requires_grad=True)
b1 = torch.randn(hidden_dim, requires_grad=True)
W2 = torch.randn((hidden_dim, 27), requires_grad=True)
b2 = torch.randn(27, requires_grad=True)


for i in range(2000):

    ix = torch.randint(0, X_train.shape[0], (32,))

    X_batch = X_train[ix]
    Y_batch = Y_train[ix]

    embedding = embed_table[X_batch]

    h = embedding.flatten(1, 2) @ W1 + b1
    h_tanh = h.tanh()

    logits = h_tanh @ W2 + b2

    loss = F.cross_entropy(logits, Y_batch)

    for p in [W1, b1, W2, b2, embed_table]:
        p.grad = None

    loss.backward()

    W1.data += -0.5 * W1.grad
    b1.data += -0.5 * b1.grad
    W2.data += -0.5 * W2.grad
    b2.data += -0.5 * b2.grad
    embed_table.data += -0.5 * embed_table.grad


with torch.no_grad():

    embedding = embed_table[X_dev]

    h = embedding.flatten(1, 2) @ W1 + b1
    h_tanh = h.tanh()

    logits = h_tanh @ W2 + b2

    dev_loss = F.cross_entropy(logits, Y_dev)


print("2D dev loss:", dev_loss.item())


learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]

results = []

for lr in learning_rates:

    embed_table = torch.randn((27, 2), requires_grad=True)
    W1 = torch.randn((6, 100), requires_grad=True)
    b1 = torch.randn(100, requires_grad=True)
    W2 = torch.randn((100, 27), requires_grad=True)
    b2 = torch.randn(27, requires_grad=True)

    for i in range(1000):

        ix = torch.randint(0, X_train.shape[0], (32,))

        X_batch = X_train[ix]
        Y_batch = Y_train[ix]

        embedding = embed_table[X_batch]

        h = embedding.flatten(1, 2) @ W1 + b1
        h_tanh = h.tanh()

        logits = h_tanh @ W2 + b2

        loss = F.cross_entropy(logits, Y_batch)

        for p in [W1, b1, W2, b2, embed_table]:
            p.grad = None

        loss.backward()

        W1.data += -lr * W1.grad
        b1.data += -lr * b1.grad
        W2.data += -lr * W2.grad
        b2.data += -lr * b2.grad
        embed_table.data += -lr * embed_table.grad

    with torch.no_grad():

        embedding = embed_table[X_dev]

        h = embedding.flatten(1, 2) @ W1 + b1
        h_tanh = h.tanh()

        logits = h_tanh @ W2 + b2

        dev_loss = F.cross_entropy(logits, Y_dev)

    results.append((lr, dev_loss.item()))


print("learning rate results:")
print(results)


embed_dim = 10
hidden_dim = 200

embed_table = torch.randn((27, embed_dim), requires_grad=True)
W1 = torch.randn((3 * embed_dim, hidden_dim), requires_grad=True)
b1 = torch.randn(hidden_dim, requires_grad=True)
W2 = torch.randn((hidden_dim, 27), requires_grad=True)
b2 = torch.randn(27, requires_grad=True)


for i in range(2000):

    ix = torch.randint(0, X_train.shape[0], (32,))

    X_batch = X_train[ix]
    Y_batch = Y_train[ix]

    embedding = embed_table[X_batch]

    h = embedding.flatten(1, 2) @ W1 + b1
    h_tanh = h.tanh()

    logits = h_tanh @ W2 + b2

    loss = F.cross_entropy(logits, Y_batch)

    for p in [W1, b1, W2, b2, embed_table]:
        p.grad = None

    loss.backward()

    W1.data += -0.1 * W1.grad
    b1.data += -0.1 * b1.grad
    W2.data += -0.1 * W2.grad
    b2.data += -0.1 * b2.grad
    embed_table.data += -0.1 * embed_table.grad


with torch.no_grad():

    embedding = embed_table[X_dev]

    h = embedding.flatten(1, 2) @ W1 + b1
    h_tanh = h.tanh()

    logits = h_tanh @ W2 + b2

    dev_loss = F.cross_entropy(logits, Y_dev)


print("10D / 200 hidden dev loss:", dev_loss.item())


with torch.no_grad():

    embedding = embed_table[X_test]

    h = embedding.flatten(1, 2) @ W1 + b1
    h_tanh = h.tanh()

    logits = h_tanh @ W2 + b2

    test_loss = F.cross_entropy(logits, Y_test)


print("test loss:", test_loss.item())


plt.figure(figsize=(10, 10))

embedding_2d = None


embed_table_2d = torch.randn((27, 2), requires_grad=True)
W1_2d = torch.randn((6, 100), requires_grad=True)
b1_2d = torch.randn(100, requires_grad=True)
W2_2d = torch.randn((100, 27), requires_grad=True)
b2_2d = torch.randn(27, requires_grad=True)


for i in range(2000):

    ix = torch.randint(0, X_train.shape[0], (32,))

    X_batch = X_train[ix]
    Y_batch = Y_train[ix]

    embedding = embed_table_2d[X_batch]

    h = embedding.flatten(1, 2) @ W1_2d + b1_2d
    h_tanh = h.tanh()

    logits = h_tanh @ W2_2d + b2_2d

    loss = F.cross_entropy(logits, Y_batch)

    for p in [W1_2d, b1_2d, W2_2d, b2_2d, embed_table_2d]:
        p.grad = None

    loss.backward()

    W1_2d.data += -0.1 * W1_2d.grad
    b1_2d.data += -0.1 * b1_2d.grad
    W2_2d.data += -0.1 * W2_2d.grad
    b2_2d.data += -0.1 * b2_2d.grad
    embed_table_2d.data += -0.1 * embed_table_2d.grad


for i, char in enumerate(letter_list):

    x, y = embed_table_2d[i].detach()

    plt.scatter(x, y)
    plt.text(x, y, char)


plt.grid()
plt.show()


distances = torch.cdist(embed_table_2d, embed_table_2d)

for i, char in enumerate(letter_list):

    nearest = distances[i].argsort()[1:5]

    print(
        char,
        [letter_list[j] for j in nearest]
    )


for _ in range(20):

    context = [0, 0, 0]
    name = ""

    while True:

        X_context = torch.tensor([context])

        embedding = embed_table[X_context]

        h = embedding.flatten(1, 2) @ W1 + b1
        h_tanh = h.tanh()

        logits = h_tanh @ W2 + b2

        probs = F.softmax(logits, dim=1)

        ix = torch.multinomial(
            probs,
            num_samples=1
        ).item()

        if ix == 0:
            break

        name += letter_list[ix]

        context = context[1:] + [ix]

        if len(name) >= 20:
            break

    print(name)


plt.hist(
    h.detach().flatten().numpy(),
    bins=50
)

plt.show()


plt.hist(
    h_tanh.detach().flatten().numpy(),
    bins=50
)

plt.show()


embed_dim = 10
hidden_dim = 200

embed_table = torch.randn(
    (27, embed_dim),
    requires_grad=True
)

W1 = (
    torch.randn(
        (3 * embed_dim, hidden_dim)
    )
    * (5 / 3)
    / ((3 * embed_dim) ** 0.5)
).requires_grad_()

b1 = (
    torch.randn(hidden_dim) * 0.01
).requires_grad_()

W2 = (
    torch.randn(
        (hidden_dim, 27)
    ) * 0.01
).requires_grad_()

b2 = torch.zeros(
    27,
    requires_grad=True
)

embedding = embed_table[X_train[:32]]

h = embedding.flatten(1, 2) @ W1 + b1
h_tanh = h.tanh()

logits = h_tanh @ W2 + b2

initial_loss = F.cross_entropy(logits, Y_train[:32])

print("initial loss:", initial_loss.item())

plt.hist(h.detach().flatten().numpy(), bins=50)
plt.title("Before tanh")
plt.show()

plt.hist(h_tanh.detach().flatten().numpy(), bins=50)
plt.title("After tanh")
plt.show()

W1 = torch.randn((30, 200), requires_grad=True)

embed_dim = 10
hidden_dim = 200

embed_table = torch.randn((27, embed_dim),requires_grad=True)

W1 = torch.randn((3 * embed_dim, hidden_dim),requires_grad=True)
b1 = torch.randn(hidden_dim,requires_grad=True)

W2 = torch.randn((hidden_dim, 27),requires_grad=True)
b2 = torch.randn(27,requires_grad=True)

embedding = embed_table[X_train[:32]]

h = embedding.flatten(1, 2) @ W1 + b1
h_tanh = h.tanh()

logits = h_tanh @ W2 + b2

initial_loss = F.cross_entropy(logits,Y_train[:32])

print("Bad init loss:", initial_loss.item())

plt.figure(figsize=(8, 5))

plt.hist(
    h.detach().flatten().numpy(),
    bins=50
)

plt.title("Bad init before tanh")
plt.xlabel("h")
plt.ylabel("count")
plt.show()

plt.figure(figsize=(8, 5))

plt.hist(
    h_tanh.detach().flatten().numpy(),
    bins=50
)

plt.title("Bad init after tanh")
plt.xlabel("tanh(h)")
plt.ylabel("count")
plt.show()

embed_table = torch.randn(
    (27, embed_dim),
    requires_grad=True
)


W1 = (torch.randn((3 * embed_dim, hidden_dim))* (5 / 3)/ ((3 * embed_dim) ** 0.5)).requires_grad_()
b1 = (torch.randn(hidden_dim) * 0.01).requires_grad_()

W2 = (torch.randn((hidden_dim, 27)) * 0.01).requires_grad_()
b2 = torch.zeros(27,requires_grad=True)

embedding = embed_table[X_train[:32]]

h = embedding.flatten(1, 2) @ W1 + b1
h_tanh = h.tanh()

logits = h_tanh @ W2 + b2

initial_loss = F.cross_entropy(logits,Y_train[:32])

print("Kaiming init loss:", initial_loss.item())

plt.figure(figsize=(8, 5))
plt.hist(h.detach().flatten().numpy(), bins=50)
plt.title("Kaiming init before tanh")
plt.xlabel("h")
plt.ylabel("count")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(h_tanh.detach().flatten().numpy(), bins=50)
plt.title("Kaiming init after tanh")
plt.xlabel("tanh(h)")
plt.ylabel("count")
plt.show()

import torch.nn as nn

embed_dim = 10
hidden_dim = 200
batch_size = 32
steps = 5000
lr = 0.1


def train_model(use_batchnorm=False):

    embed_table = torch.randn((27, embed_dim), requires_grad=True)
    W1 = (torch.randn((3 * embed_dim, hidden_dim)) * (5 / 3) / ((3 * embed_dim) ** 0.5)).requires_grad_()
    b1 = (torch.randn(hidden_dim) * 0.01).requires_grad_()
    W2 = (torch.randn((hidden_dim, 27)) * 0.01).requires_grad_()
    b2 = torch.zeros(27, requires_grad=True)

    if use_batchnorm:
        bn = nn.BatchNorm1d(hidden_dim)
        bn.train()
    else:
        bn = None

    parameters = [embed_table, W1, b1, W2, b2]

    if use_batchnorm:
        parameters += list(bn.parameters())

    for i in range(steps):

        ix = torch.randint(0, X_train.shape[0], (batch_size,))

        X_batch = X_train[ix]
        Y_batch = Y_train[ix]

        embedding = embed_table[X_batch]

        h = embedding.flatten(1, 2) @ W1 + b1

        if use_batchnorm:
            h = bn(h)

        h = h.tanh()

        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, Y_batch)

        for p in parameters:
            p.grad = None

        loss.backward()

        for p in parameters:
            p.data += -lr * p.grad

    if use_batchnorm:
        bn.eval()

    with torch.no_grad():

        embedding = embed_table[X_dev]

        h = embedding.flatten(1, 2) @ W1 + b1

        if use_batchnorm:
            h = bn(h)

        h = h.tanh()

        logits = h @ W2 + b2

        dev_loss = F.cross_entropy(logits, Y_dev)

    return dev_loss.item()


loss_no_bn = train_model(False)
loss_bn = train_model(True)

print("without batchNorm:", loss_no_bn)
print("with batchNorm:", loss_bn)