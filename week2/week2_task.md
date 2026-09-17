# Week 02 — Backpropagation & Autograd

> **YZ50 — Week 02**

This week focuses on backpropagation, computation graphs, the chain rule, and the mechanism behind automatic differentiation.

Bu hafta backpropagation, computation graph, chain rule ve automatic differentiation'ın arkasındaki mekanizmaya odaklanıyorum.

---

## 📚 Topics / Konular

### English

This week I will learn:

* The intuition behind the chain rule
* Computation graphs and how gradients flow through them
* The backward pass
* How local derivatives are combined with upstream gradients
* How to verify analytical gradients using numerical derivatives
* How the same mechanism forms the core of PyTorch's `autograd`

### Türkçe

Bu hafta öğreneceklerim:

* Chain rule (zincir kuralı) sezgisi
* Computation graph: Her işlemin bir node olması ve gradient'in çıktıdan girdiye doğru akması
* Backward pass mantığı
* Her node'da local derivative ile upstream gradient'in çarpılması
* Analitik gradient'in numerical derivative ile doğrulanması
* PyTorch `autograd` mekanizmasının temelinde aynı prensiplerin bulunması

---

## 🎥 Resources / Kaynaklar

### 1. Andrej Karpathy — The spelled-out intro to neural networks and backpropagation

[Watch on YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)

This week the **full video** is covered. The first 19 minutes were studied during Week 01.

Bu hafta videonun **tamamı** ele alınıyor. İlk 19 dakika Week 01'de işlendi.

---

### 2. Karpathy — micrograd

[GitHub Repository](https://github.com/karpathy/micrograd)

A reference implementation to consult when needed while building the autograd mechanism from scratch.

Autograd mekanizmasını sıfırdan geliştirirken takıldığım noktalarda referans olarak kullanılacak.

---

### 3. 3Blue1Brown — What is backpropagation really doing?

[Watch on YouTube](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

Visual intuition for backpropagation and how gradients propagate through a computation.

Backpropagation'ın nasıl çalıştığını ve gradient'lerin computation graph üzerinden nasıl ilerlediğini görsel olarak anlamaya yönelik kaynak.

---

### 4. 3Blue1Brown — Backpropagation calculus

[Watch on YouTube](https://www.youtube.com/watch?v=tIeHLnjs5U8)

A calculus-focused explanation of backpropagation and the chain rule.

Backpropagation ve chain rule'un calculus açısından açıklanmasına odaklanıyor.

---

# 🧪 Tasks / Görevler

## 1. Build a `Value` Class

Create your own `Value` class, starting with addition and multiplication.

Each new `Value` should keep track of:

* The `Value` objects that produced it
* The operation that produced it

Optionally, visualize the computation graph using Graphviz.

**Video:** `19:09 – 32:10`

**Goal:** Understand how a computation can be represented as a graph and how each operation becomes a node.

**Amaç:** Hesaplamaların bir graph olarak nasıl temsil edildiğini ve her işlemin computation graph içerisinde nasıl bir node oluşturduğunu anlamak.

**File:** `01_value.py`

---

## 2. Calculate Gradients Manually

Before automating the process, manually calculate the gradients for the two examples from the video:

1. A simple mathematical expression
2. A single neuron

Add `tanh` to the `Value` class for the neuron example.

Use these examples to understand the chain rule step by step.

**Video:** `32:10 – 51:10`
**Neuron:** `52:52 – 1:09:02`

**Goal:** Build an intuition for how gradients are calculated by applying the chain rule through a computation graph.

**Amaç:** Chain rule'u computation graph üzerinde adım adım uygulayarak gradient'lerin nasıl hesaplandığını içselleştirmek.

**File:** `02_manual_gradients.py`

---

## 3. Implement `backward()`

Implement an automatic `backward()` method.

The method should:

1. Set the output node's gradient to `1`
2. Traverse the nodes in reverse topological order
3. Apply the chain rule at each node
4. Accumulate gradients when a variable is used in multiple places

Gradients must be **accumulated, not overwritten**.

**Video:** `1:09:02 – 1:27:05`

**Goal:** Turn the manual backpropagation process into an automatic mechanism.

**Amaç:** Elle yaptığım backpropagation işlemini otomatik hale getirmek ve computation graph üzerinden gradient hesaplayabilen bir `backward()` mekanizması oluşturmak.

**File:** `03_backward.py`

---

## 4. Decompose `tanh` and Verify the Gradients

Break the `tanh` operation down into simpler operations:

* `exp`
* division
* `pow`

Show that the resulting gradients are equivalent.

Then verify the implementation using three different methods:

1. `backward()`
2. The numerical derivative implementation from Week 01
3. PyTorch

Use the same expression and the same values for all three methods and verify that the gradients match.

**Video:** `1:27:05 – 1:43:55`

**Goal:** Understand that complex operations can be built from simpler differentiable operations and verify that the custom autograd implementation is correct.

**Amaç:** Daha karmaşık bir işlemin daha basit differentiable operasyonlara ayrılabileceğini görmek ve kendi gradient hesaplamamı numerical derivative ve PyTorch ile doğrulamak.

**File:** `04_gradient_check.py`

---

## 5. Build Neuron, Layer, and MLP

Following the structure from the video, implement:

* `Neuron`
* `Layer`
* `MLP`

Collect all model parameters into a single list.

Train the MLP using the small dataset from the video and show that the loss decreases step by step.

Remember to **zero the gradients before each update**.

> A missing gradient reset is the famous bug demonstrated in the video.

**Video:** `1:43:55 – 2:14:03`

**Goal:** Combine the previous concepts into a small neural network that can actually learn from data.

**Amaç:** Önceki haftalarda ve bu hafta öğrenilen kavramları birleştirerek gerçekten veri üzerinden öğrenebilen küçük bir MLP oluşturmak.

**File:** `05_mlp.py`

---

# 🧠 Learning Notes / Öğrenme Notları

This section will contain my own notes, observations, and conclusions from the week's materials and experiments.

Bu bölümde hafta boyunca öğrendiğim kavramları, yaptığım deneyleri ve çıkardığım sonuçları kendi cümlelerimle paylaşacağım.

Questions to explore:

* What is the chain rule actually doing?
* Why does the gradient flow from output to input?
* What is an upstream gradient?
* What is a local derivative?
* Why do gradients need to be accumulated?
* What is a topological ordering?
* Why do we need reverse topological ordering for backpropagation?
* How does `backward()` automate the manual chain rule?
* Why does numerical differentiation help verify gradients?
* How is this related to PyTorch's `autograd`?
* Why must gradients be reset before each parameter update?

---

# 🔬 Gradient Verification

One of the main goals of this week is to verify that the manually implemented backward pass produces correct gradients.

The same expression will be evaluated using:

```text
Custom backward()
        ↓
Numerical derivative
        ↓
PyTorch autograd
```

The resulting gradients should match within a small numerical tolerance.

---

# 📁 Structure / Dosya Yapısı

```text
week-02-backpropagation/
│
├── README.md
├── notes.md
│
├── 01_value.py
├── 02_manual_gradients.py
├── 03_backward.py
├── 04_gradient_check.py
└── 05_mlp.py
```

---

# 🎯 Outcome / Hafta Sonu Hedefi

### English

By the end of this week, I should be able to implement a small automatic differentiation engine from scratch, understand how backpropagation applies the chain rule through a computation graph, verify gradients numerically, and use the resulting system to train a small MLP.

### Türkçe

Bu haftanın sonunda sıfırdan küçük bir automatic differentiation engine oluşturabilmeli, backpropagation'ın computation graph üzerinde chain rule'u nasıl uyguladığını anlayabilmeli, gradient'leri numerical derivative ve PyTorch ile doğrulayabilmeli ve oluşturduğum sistem ile küçük bir MLP'yi eğitebilmeliyim.
