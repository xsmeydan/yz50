# # read names.txt

# with open('names.txt', 'r') as file:
#     names = file.readlines()
#     names = [name.strip() for name in names]

# print(len(names))
# print(names[:10])
# print(max(len(name) for name in names))

# #bigram
# for name in names:
#     name = '.' + name + '.'
#     print(name)
#     for i in range(len(name) - 1):
#         bigram = name[i:i+2]
#         print(bigram)

# counts = {}
# for name in names:
#     name = '.' + name + '.'
#     for i in range(len(name) - 1):
#         bigram = name[i:i+2]
#         counts[bigram] = counts.get(bigram, 0) + 1

# print(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])

# letterstoi = {}
# letteritos = {}
# letter_list = '.abcdefghijklmnopqrstuvwxyz'
# for i in range(len(letter_list)):
#     letterstoi[letter_list[i]] = i
#     letteritos[i] = letter_list[i]

# import torch

# N = torch.zeros((27, 27))
# print(N.shape)
# print(N[0, 0])
# for name in names:
#     name = '.' + name + '.'
#     for i in range(len(name) - 1):
#         bigram = name[i:i+2]
#         ix1 = letterstoi[bigram[0]]
#         ix2 = letterstoi[bigram[1]]
#         N[ix1, ix2] += 1

# print(N.sum())

# import matplotlib.pyplot as plt
# # plt.imshow(N)
# # plt.xticks(range(27), letter_list)
# # plt.yticks(range(27), letter_list)
# # plt.show()

# plt.imshow(N)
# plt.xticks(range(27), letter_list)
# plt.yticks(range(27), letter_list)
# for i in range(27):
#     for j in range(27):
#         plt.text(j, i, N[i, j].item(), ha='center', va='center', color='w')
# # plt.show()

# P = N / N.sum(1, keepdim=True)
# print(P.shape)
# print(P[0, :].sum())

# g = torch.Generator().manual_seed(2147483647)
# ix = torch.multinomial(P[0], num_samples=1, replacement=True, generator=g)

# print(ix)

# ix = 0
# name = ''
# while True:
#     ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
#     print(letteritos[ix])
#     if ix == 0:
#         break
#     name += letteritos[ix]

# print(name)

# for _ in range(20):
#     ix = 0
#     name = ''
#     while True:
#         ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item()
#         if ix == 0:
#             break
#         name += letteritos[ix]
#     print(name)

# print(P[0, 1])
# print(torch.log(P[0, 1]))
# print(P.min())

# P = (N + 1) / (N + 1).sum(1, keepdim=True)
# print(P.min())
# print(P.sum(1))

# nll = (N * -torch.log(P)).sum() / N.sum()
# print(nll)

# xs=[]
# ys=[]
# for name in names:
#     name = '.' + name + '.'
#     for i in range(len(name) - 1):
#         bigram = name[i:i+2]
#         ix1 = letterstoi[bigram[0]]
#         ix2 = letterstoi[bigram[1]]
#         xs.append(ix1)
#         ys.append(ix2)

# print(xs[:10])
# print(ys[:10])

# xs = torch.tensor(xs)
# ys = torch.tensor(ys)

# print(xs.shape)
# print(ys.shape)

# import torch.nn.functional as F

# xenc = F.one_hot(xs, num_classes=27).float()
# print(xenc.shape)

# W = torch.randn((27, 27), requires_grad=True)
# print(W.shape)

# logits = xenc @ W
# print(logits.shape)
# print(logits[0])

# counts = logits.exp()
# probs = counts / counts.sum(1, keepdim=True)
# print(probs.shape)
# print(probs[0])
# print(probs[0].sum())

# print(probs[0, ys[0]])
# nll = -torch.log(probs[0, ys[0]])
# print(nll)

# p = probs[torch.arange(len(ys)), ys]
# print(p.shape)
# print(p[:10])

# nlls = -torch.log(p)
# loss = nlls.mean()

# print(loss)

# W.grad = None
# loss.backward()
# print(W.grad.shape)

# W.data += -0.1 * W.grad
# logits = xenc @ W
# counts = logits.exp()
# probs = counts / counts.sum(1, keepdim=True)
# p = probs[torch.arange(len(ys)), ys]
# nlls = -torch.log(p)
# loss = nlls.mean()
# print(loss)

# for k in range(100):
#     W.grad = None
#     logits = xenc @ W
#     counts = logits.exp()
#     probs = counts / counts.sum(1, keepdim=True)
#     p = probs[torch.arange(len(ys)), ys]
#     nlls = -torch.log(p)
#     loss = nlls.mean()
#     loss.backward()
#     W.data += -50 * W.grad
#     # if k % 10 == 0:
#     #     print(k, loss.item())

# print(N[0])
# print(W[0])

# P_nn = W.exp()
# P_nn = P_nn / P_nn.sum(1, keepdim=True)
# print(P_nn[0])

# P_count = (N + 1) / (N + 1).sum(1, keepdim=True)
# print(P_count[0])

import torch
import torch.nn.functional as F

with open('isimler.txt', 'r') as f:
    names = f.readlines()
    names = [name.strip().lower() for name in names]

print(len(names))
names = names[1:]
print(names[:10])

letterstoi = {}
letteritos = {}
letter_list = '.abcçdefgğhıijklmnoöprsştuüvyz'
for i in range(len(letter_list)):
    letterstoi[letter_list[i]] = i
    letteritos[i] = letter_list[i]

chars = set(''.join(names))
print(sorted(chars))


N = torch.zeros((30, 30))
print(N.shape)
print(N[0, 0])

for name in names:
    if len(name) > 0:
        name = '.' + name + '.'
        for i in range(len(name) - 1):
            bigram = name[i:i+2]
            if bigram[0] not in letterstoi or bigram[1] not in letterstoi:
                continue

            ix1 = letterstoi[bigram[0]]
            ix2 = letterstoi[bigram[1]]
            N[ix1, ix2] += 1

print(N.sum())

P = (N + 1) / (N + 1).sum(1, keepdim=True)
print(P.shape)
print(P.sum(1))
print(P.min())

nll = (N * -torch.log(P)).sum() / N.sum()
print(nll)

xs = []
ys = []

for name in names:
    name = '.' + name + '.'
    for i in range(len(name) - 1):
        bigram = name[i:i+2]
        if bigram[0] not in letterstoi or bigram[1] not in letterstoi:
            continue
        ix1 = letterstoi[bigram[0]]
        ix2 = letterstoi[bigram[1]]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs.shape)
print(ys.shape)

xenc = F.one_hot(xs, num_classes=30).float()
print(xenc.shape)

W = torch.randn((30, 30), requires_grad=True)
print(W.shape)

logits = xenc @ W
print(logits.shape)

counts = logits.exp()
probs = counts / counts.sum(1, keepdim=True)
print(probs.shape)
print(probs[0])

for i in range(1000):
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)

    p = probs[torch.arange(len(ys)), ys]
    loss = -torch.log(p).mean()

    W.grad = None
    loss.backward()

    W.data += -0.5 * W.grad

    if i % 10 == 0:
        print(i, loss.item())

g = torch.Generator().manual_seed(2147483647)

for _ in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(letteritos[ix])

        if ix == 0:
            break

    print(''.join(out))

g = torch.Generator().manual_seed(214748364)
P_nn = W.exp()
P_nn = P_nn / P_nn.sum(1, keepdim=True)
print(P_nn[0])

for _ in range(10):
    out = []
    ix = 0
    while True:
        p = P_nn[ix]
        ix = torch.multinomial(
            p,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()

        out.append(letteritos[ix])
        if ix == 0:
            break

    print(''.join(out))