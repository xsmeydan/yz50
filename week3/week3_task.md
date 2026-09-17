# Week 03 — Bigram Language Model

> **YZ50 — Week 03**

This week introduces language modeling through a character-level bigram model. I will build the same model in two different ways: first using counts and probability distributions, and then using a simple neural network trained with gradient descent.

Bu hafta karakter seviyesinde bir bigram language model oluşturarak language modeling kavramına giriş yapıyorum. Aynı modeli önce sayım ve olasılık dağılımlarıyla, ardından gradient descent ile eğitilen basit bir neural network kullanarak kuracağım.

---

## 📚 Topics / Konular

### English

This week I will learn:

* Character-level bigram language models
* Turning count tables into probability distributions
* Sampling new sequences from a language model
* Negative log likelihood (NLL)
* Why log, why negative, and why the mean?
* One-hot encoding
* Logits and softmax
* Training a bigram model with gradient descent
* The connection between the counting-based and neural-network-based models
* How the `backward()` mechanism from Week 02 is used by PyTorch tensors

### Türkçe

Bu hafta öğreneceklerim:

* Karakter seviyesinde bigram language model
* Sayım tablolarını olasılık dağılımlarına dönüştürme
* Language model'den yeni örnekler üretme (sampling)
* Negative log likelihood (NLL)
* Neden log, neden negatif ve neden ortalama?
* One-hot encoding
* Logits ve softmax
* Gradient descent ile bigram model eğitmek
* Sayım tabanlı model ile neural network tabanlı model arasındaki bağlantı
* Week 02'de yazdığım `backward()` mekanizmasının PyTorch tensor'larında nasıl kullanıldığı

---

## 🎥 Resources / Kaynaklar

### 1. Andrej Karpathy — The spelled-out intro to language modeling: building makemore

[Watch on YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo)

The main resource for this week's exercises.

Bu haftanın görevleri için ana kaynak.

---

### 2. Karpathy — makemore

[GitHub Repository](https://github.com/karpathy/makemore)

Reference implementation for the exercises. The English `names.txt` dataset used in the first tasks is also available here.

Görevlerde referans olarak kullanılabilir. İlk görevlerde kullanılacak İngilizce `names.txt` veri seti de burada bulunuyor.

---

### 3. PyTorch — Broadcasting

[PyTorch Broadcasting Documentation](https://pytorch.org/docs/stable/notes/broadcasting.html)

Useful for understanding tensor operations in Task 2, especially the `keepdim` issue.

Özellikle Task 2'deki tensor işlemlerini ve videodaki `keepdim` kaynaklı hatayı anlamak için kullanılacak.

---

# 🧪 Tasks / Görevler

## 1. Count Character Bigrams

Start with Karpathy's English `names.txt` dataset.

Count character bigrams in two ways:

1. Using a Python dictionary
2. Using a `27 × 27` PyTorch tensor

Visualize the resulting count table similar to the visualization in the video.

**Video:** `3:03 – 24:02`

**Goal:** Understand how a character-level language model can be represented using simple counts.

**Amaç:** Karakter seviyesinde bir language model'in en temel haliyle karakter çiftlerinin sayımlarından nasıl oluşturulabileceğini anlamak.

**File:** `01_bigram_counts.py`

---

## 2. Convert Counts to Probabilities & Sample

Convert each row of the count table into a probability distribution.

Use these distributions to sample new names from the model.

Pay special attention to PyTorch broadcasting and the `keepdim` issue demonstrated in the video.

A broadcasting mistake can silently produce an incorrect model, so verify the tensor shapes carefully.

**Video:** `24:02 – 50:14`

**Goal:** Understand how observed counts become conditional probability distributions and how those probabilities can be used to generate new samples.

**Amaç:** Karakter sayımlarının koşullu olasılık dağılımlarına nasıl dönüştürüldüğünü ve bu dağılımlardan yeni isimlerin nasıl üretildiğini anlamak.

**File:** `02_bigram_sampling.py`

---

## 3. Negative Log Likelihood

Calculate the Negative Log Likelihood (NLL) of the bigram model.

Add fake counts for smoothing and observe how this affects the probabilities and loss.

Explain in your own words:

* Why do we use logarithms?
* Why do we take the negative?
* Why do we take the mean?

These explanations will also be included in the submission video.

**Video:** `50:14 – 1:02:57`

**Goal:** Understand how NLL measures the quality of a language model using a single number.

**Amaç:** NLL'in bir language model'in kalitesini tek bir değerle nasıl ölçtüğünü ve log, negative ve mean kullanmamızın nedenlerini anlamak.

**File:** `03_negative_log_likelihood.py`

---

## 4. Build the Bigram Model with a Neural Network

Build the same bigram model using a single-layer neural network.

The model should contain:

* One-hot encoded input
* A `27 × 27` weight matrix
* Logits
* Softmax
* Negative log likelihood loss
* Gradient descent

Train the model and show that its loss approaches the loss of the counting-based bigram model from Task 3.

The `backward()` mechanism used here is conceptually the same mechanism implemented manually with `micrograd` during Week 02. PyTorch applies the same principle to tensors.

**Video:** `1:02:57 – 1:54:31`

**Goal:** Understand that the same bigram model can be expressed either as a probability table or as a neural network.

**Amaç:** Aynı bigram modelinin hem doğrudan probability/count tablosuyla hem de neural network + gradient descent kullanılarak kurulabileceğini görmek.

**File:** `04_neural_bigram.py`

---

# 🇹🇷 5. Turkish Character-Level Bigram Model

Find an open-source Turkish name dataset and run both models on the new dataset.

Extend the character vocabulary to support Turkish characters:

```text
ç ğ ı ö ş ü
```

Run both versions of the model:

1. Count-based bigram model
2. Neural-network-based bigram model

Show:

* Example generated names
* Model loss
* Comparison between the two approaches

Document any observations about how the generated names differ from the English model.

**Goal:** Extend the model beyond the original dataset and apply the concepts to Turkish text.

**Amaç:** Modeli İngilizce veri setinin dışına çıkararak Türkçe isimler üzerinde çalıştırmak ve karakter setini Türkçe karakterleri destekleyecek şekilde genişletmek.

**File:** `05_turkish_bigram.py`

---

# ⭐ Optional — Trigram Language Model

For an additional challenge, extend the model from bigram to trigram.

Instead of looking at one previous character, the model should use the previous **two characters** to predict the next character.

### Tasks

* Convert the bigram model into a trigram model
* Split the dataset into:

  * `80%` train
  * `10%` dev
  * `10%` test
* Tune the smoothing strength using the dev loss
* Compare bigram and trigram losses
* Generate names with both models
* Observe how the generated names change

This exercise is not implemented in the video. It is given by Karpathy as an exercise at the end of the video.

**Goal:** Understand how increasing the context size changes a character-level language model.

**Amaç:** Modelin baktığı context'i bir karakterden iki karaktere çıkarmanın language model'in loss'u ve ürettiği isimler üzerindeki etkisini gözlemlemek.

**File:** `06_trigram.py`

---

# 🧠 Learning Notes / Öğrenme Notları

This section will contain my own notes, observations, and conclusions from the week's materials and experiments.

Bu bölümde hafta boyunca öğrendiğim kavramları, yaptığım deneyleri ve çıkardığım sonuçları kendi cümlelerimle paylaşacağım.

Questions to explore:

* What does a bigram language model actually learn?
* What does `P(next character | current character)` represent?
* Why does the count table form a probability distribution?
* What does sampling from the model mean?
* Why can a model with higher probabilities for correct characters have a lower NLL?
* Why do we use logarithms in NLL?
* What is the relationship between logits and probabilities?
* Why do we need softmax?
* How does one-hot encoding represent a character?
* How can a weight matrix represent the same information as a count table?
* How does gradient descent learn the weights?
* How is PyTorch's `autograd` connected to the `backward()` implementation from Week 02?
* What changes when moving from bigram to trigram?

---

# 🔬 Model Comparison

One of the main goals of this week is to show that the same bigram model can be represented in two different ways:

```text
                    BIGRAM MODEL
                         │
              ┌──────────┴──────────┐
              │                     │
        Count-based             Neural Network
              │                     │
       Count table              One-hot input
              │                     │
     Probability table            Weights
              │                     │
           Sampling             Softmax
              │                     │
             NLL                   NLL
              │                     │
              └──────────┬──────────┘
                         │
                  Compare results
```

The neural-network version should approach the loss achieved by the count-based version.

---

# 📁 Structure / Dosya Yapısı

```text
week-03-bigram-language-model/
│
├── README.md
├── notes.md
│
├── 01_bigram_counts.py
├── 02_bigram_sampling.py
├── 03_negative_log_likelihood.py
├── 04_neural_bigram.py
├── 05_turkish_bigram.py
└── 06_trigram.py
```

---

# 🎯 Outcome / Hafta Sonu Hedefi

### English

By the end of this week, I should be able to build a character-level bigram language model from scratch using both counting and gradient descent, understand NLL as a language-model objective, generate samples from the model, and explain the connection between the two implementations.

As an extension, I should be able to adapt the model to Turkish characters and experiment with a larger context using a trigram model.

### Türkçe

Bu haftanın sonunda karakter seviyesinde bir bigram language model'i hem sayım yöntemiyle hem de gradient descent kullanarak sıfırdan oluşturabilmeli, NLL'in language model eğitimindeki rolünü anlayabilmeli, modelden yeni örnekler üretebilmeli ve iki farklı yaklaşım arasındaki bağlantıyı açıklayabilmeliyim.

Ek olarak modeli Türkçe karakterleri destekleyecek şekilde genişletmeli ve trigram modeliyle daha büyük bir context kullanmayı deneyebilmeliyim.
