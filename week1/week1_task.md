
# Week 01 — Neural Networks

> **YZ50 — Week 01**

This week focuses on the fundamental ideas behind neural networks and how neural networks learn.

Bu hafta sinir ağlarının temel kavramlarını ve sinir ağlarının nasıl öğrendiğini anlamaya odaklanıyorum.

---

## 📚 Topics / Konular

### English

This week I will learn:

* Neuron, layer, parameter, and loss
* How a forward pass works
* What it means for a model to learn
* The intuition behind gradient descent
* Introduction to the language model concept

### Türkçe

Bu hafta öğreneceklerim:

* Nöron, katman, parametre ve loss kavramları
* Forward pass mantığı
* Bir modelin öğrenmesinin ne anlama geldiği
* Gradient descent'in temel sezgisi
* Language model kavramına giriş

---

## 🎥 Resources / Kaynaklar

### 1. 3Blue1Brown — But what is a Neural Network?

[Watch on YouTube](https://www.youtube.com/watch?v=aircAruvnKk)

Introduction to neural networks, neurons, layers, weights, biases, and the general structure of a neural network.

Sinir ağlarına, nöronlara, katmanlara, weight ve bias kavramlarına ve bir sinir ağının genel yapısına giriş.

---

### 2. 3Blue1Brown — Gradient descent, how neural networks learn

[Watch on YouTube](https://www.youtube.com/watch?v=IHZwWFHWa-w)

Understanding how parameters are adjusted to reduce the loss and the intuition behind gradient descent.

Parametrelerin loss'u azaltacak şekilde nasıl güncellendiğini ve gradient descent'in arkasındaki temel sezgiyi anlamaya yönelik kaynak.

---

### 3. Andrej Karpathy — The spelled-out intro to neural networks

[Watch on YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0)

The first **19 minutes**, focusing on numerical derivatives and the foundations needed to understand how neural networks can be trained.

İlk **19 dakika**, özellikle numerical derivative ve neural network'lerin nasıl eğitildiğini anlamak için gereken temel kavramlara odaklanıyor.

---

# 🧪 Tasks / Görevler

## 1. Single Neuron Forward Pass

Implement a single neuron forward pass in Python **without using machine learning libraries**.

**Goal:** Understand how inputs, weights, bias, and an activation function are combined to produce an output.

**Amaç:** Input, weight, bias ve activation function kullanılarak bir nöronun nasıl output ürettiğini anlamak.

**File:** `01_single_neuron.py`

---

## 2. Small Neural Network Layer

Extend the previous implementation to create a small layer containing multiple neurons.

Implement the forward pass for the entire layer.

**Goal:** Understand how multiple neurons work together as a layer.

**Amaç:** Birden fazla nöronun aynı katman içerisinde nasıl birlikte çalıştığını ve forward pass'in katman seviyesinde nasıl gerçekleştiğini anlamak.

**File:** `02_layer_forward_pass.py`

---

## 3. Loss Function

Implement a simple loss function from scratch.

Use it to measure the difference between the model's prediction and the expected output.

**Goal:** Understand why a loss function is needed during training.

**Amaç:** Modelin tahmini ile beklenen sonuç arasındaki farkı ölçmek için loss function'ın neden gerekli olduğunu anlamak.

**File:** `03_loss_function.py`

---

## 4. Observe the Loss

Manually change the model parameters and observe how the loss changes.

Plot the relationship between the parameter values and the resulting loss.

**Goal:** Build an intuition for how model parameters affect the loss.

**Amaç:** Model parametrelerinin değişmesinin loss üzerindeki etkisini gözlemlemek ve modelin hangi parametre değerlerinde daha düşük loss ürettiğini görmek.

**File:** `04_loss_curve.py`

---

## 5. Numerical Derivative & Gradient Descent

Use a numerical derivative to estimate the gradient of the loss with respect to a parameter.

Then implement a simple gradient descent loop that updates the parameter in small steps to reduce the loss.

**Goal:** Understand how a model can gradually adjust its parameters to minimize the loss.

**Amaç:** Modelin loss'u azaltmak için parametrelerini küçük adımlarla nasıl güncellediğini ve gradient descent'in bu süreçte nasıl kullanıldığını anlamak.

**File:** `05_gradient_descent.py`

---

# 🧠 Learning Notes / Öğrenme Notları

This section will contain my own notes, observations, and conclusions from the week's materials and experiments.

Bu bölümde hafta boyunca öğrendiğim kavramları, yaptığım gözlemleri ve çıkardığım sonuçları kendi cümlelerimle paylaşacağım.

* What is a neuron?
* What happens during a forward pass?
* What are parameters?
* Why do we need a loss function?
* How does changing a parameter affect the loss?
* What does a numerical derivative tell us?
* Why does gradient descent reduce the loss?
* How does this connect to language models?

---

# 🎯 Outcome / Hafta Sonu Hedefi

### English

By the end of this week, I should be able to implement the basic building blocks of a neural network from scratch and explain how parameters, loss, numerical derivatives, and gradient descent are connected.

### Türkçe

Bu haftanın sonunda bir sinir ağının temel yapı taşlarını kütüphane kullanmadan oluşturabilmeli ve **parametre, loss, numerical derivative ve gradient descent** kavramlarının birbirleriyle nasıl bağlantılı olduğunu açıklayabilmeliyim.
