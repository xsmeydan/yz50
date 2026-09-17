
# Week 04 — MLP Language Model

> **YZ50 — Week 04**

This week focuses on building a neural language model with learned embeddings and a larger context window. I will also explore minibatch training, dataset splitting, initialization, activation behavior, and BatchNorm.

Bu hafta öğrenilen embedding'leri ve daha geniş bir context window'u kullanan neural language model oluşturuyorum. Ayrıca minibatch training, dataset splitting, weight initialization, activation davranışı ve BatchNorm konularını inceleyeceğim.

---

## 📚 Topics / Konular

### English

This week I will learn:

* Embeddings as learned vector representations
* Context windows and predicting from multiple previous characters
* Minibatch training
* Learning rate selection
* Train / dev / test dataset splits
* Tanh saturation and vanishing gradients
* Kaiming initialization
* Batch normalization
* Comparing model capacity and validation performance
* Visualizing learned embeddings

### Türkçe

Bu hafta öğreneceklerim:

* Harfleri one-hot yerine öğrenilen embedding vektörleriyle temsil etmek
* Context window kullanarak birden fazla önceki harften sonraki harfi tahmin etmek
* Minibatch ile eğitim
* Learning rate seçimi
* Train / dev / test veri ayrımı
* Tanh saturation ve vanishing gradient
* Kaiming initialization
* Batch normalization
* Model kapasitesinin ve validation performansının karşılaştırılması
* Öğrenilen embedding'lerin görselleştirilmesi

---

## 🎥 Resources / Kaynaklar

### 1. Andrej Karpathy — Building makemore Part 2: MLP

[Watch on YouTube](https://www.youtube.com/watch?v=TCH_1BHY58I)

The main resource for building the MLP language model.

MLP language model'in oluşturulması için ana kaynak.

---

### 2. Andrej Karpathy — Building makemore Part 3: Activations & Gradients, BatchNorm

[Watch on YouTube](https://www.youtube.com/watch?v=P6sfmUTpUmc)

Focuses on activation behavior, gradients, initialization, and BatchNorm.

Activation'ların davranışı, gradient'ler, initialization ve BatchNorm konularına odaklanıyor.

---

### 3. Bengio et al. (2003) — A Neural Probabilistic Language Model

[Read the paper](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

The paper behind the neural probabilistic language model explored in Part 2.

Part 2'de ele alınan neural probabilistic language model'in temelindeki makale.

---

### 4. Karpathy — makemore

[GitHub Repository](https://github.com/karpathy/makemore)

Reference implementation for the exercises.

Görevlerde gerektiğinde referans olarak kullanılacak.

---

# 🧪 Tasks / Görevler

## 1. Build a Context Dataset and Embeddings

Start with the English `names.txt` dataset.

Build a dataset where the previous **three characters** are used as context to predict the next character.

```text
X → indices of 3 previous characters
Y → index of the next character
```

Create a `27 × 2` embedding table and use indexing to retrieve the embedding vectors for each input character.

**Video:** `Part 2 — 9:03 – 18:35`

**Goal:** Understand how characters can be represented by learned low-dimensional vectors instead of one-hot vectors.

**Amaç:** Karakterlerin one-hot vektörler yerine öğrenilen düşük boyutlu embedding vektörleriyle nasıl temsil edildiğini anlamak.

**File:** `01_embeddings.py`

---

## 2. Build the MLP

Build the hidden and output layers.

The architecture should follow:

```text
3 character context
        ↓
Embedding lookup
        ↓
Flatten embeddings
        ↓
W1 + b1
        ↓
tanh
        ↓
W2 + b2
        ↓
logits
        ↓
softmax / cross-entropy
```

First calculate the loss manually, as in the previous week.

Then calculate the same loss using:

```python
F.cross_entropy(...)
```

Verify that the results match and understand why the optimized PyTorch implementation is preferred.

**Video:** `18:35 – 37:56`

**Goal:** Build the complete forward pass of a neural language model.

**Amaç:** Embedding'den başlayarak hidden layer ve output layer üzerinden logits üreten tam bir neural language model forward pass'i oluşturmak.

**File:** `02_mlp_forward.py`

---

## 3. Train with Minibatches

Build the training loop.

Start by intentionally overfitting a single minibatch.

Then train the model using minibatches over the complete training dataset.

Experiment with different learning rates and choose a reasonable value based on the results.

Split the dataset into:

```text
80% → train
10% → dev
10% → test
```

Report the model's loss on the dev set.

**Video:** `37:56 – 1:00:49`

**Goal:** Understand minibatch training, learning-rate selection, and why separate datasets are needed for training and evaluation.

**Amaç:** Minibatch training ve learning rate seçimini anlamak; train, dev ve test setlerinin neden ayrıldığını görmek.

**File:** `03_training.py`

---

## 4. Experiment with Model Size and Embeddings

Increase the size of:

* The hidden layer
* The embedding dimension

Observe how the dev loss changes as model capacity increases.

Visualize the learned embeddings in 2D.

Investigate:

* Which characters end up close to each other?
* Do similar characters form groups?
* What patterns emerge in the embedding space?

Generate names from the model and compare them with the names generated by the bigram model from Week 03.

**Video:** `1:00:49 – 1:13:24`

**Goal:** Understand how model capacity and learned representations affect language-model performance.

**Amaç:** Model kapasitesinin ve öğrenilen embedding'lerin language model performansını nasıl etkilediğini görmek.

**File:** `04_model_experiments.py`

---

## 5. Tanh Saturation & Kaiming Initialization

Move on to Part 3.

Investigate why the model's initial loss can be much higher than expected.

Examine why `tanh` saturates when activations become concentrated around `-1` and `1`.

Reproduce the activation / gradient histograms from the video.

Then apply Kaiming initialization to scale the weights appropriately.

Show how this improves:

* Initial loss
* Activation distribution
* Gradient flow

**Video:** `Part 3 — 4:19 – 40:40`

**Goal:** Understand how initialization affects activations and gradients, and why poor initialization can make training difficult.

**Amaç:** Weight initialization'ın activation ve gradient'leri nasıl etkilediğini ve kötü initialization'ın eğitimi neden zorlaştırabildiğini anlamak.

**File:** `05_initialization.py`

---

## 6. Add BatchNorm

Add a BatchNorm layer after the hidden layer.

During training, use statistics calculated from the current batch.

During inference, use the running mean and variance.

Train two versions of the model:

1. Without BatchNorm
2. With BatchNorm

Compare their dev losses and observe the differences in training behavior.

**Video:** `40:40 – 1:04:50`

**Goal:** Understand what BatchNorm does to activations and how it affects training.

**Amaç:** BatchNorm'un activation'ları nasıl normalize ettiğini ve eğitim sürecini nasıl etkilediğini anlamak.

**File:** `06_batch_norm.py`

---

## 7. Train the Model on Turkish Names

Use the Turkish name dataset from Week 03.

Train the same MLP language model on the Turkish dataset, including the extended Turkish character vocabulary.

Generate example names and report the dev loss.

Compare the results side by side with the Turkish bigram model from Week 03.

### Compare

```text
Turkish Bigram
      vs.
Turkish MLP
```

Compare:

* Dev loss
* Generated names
* Context size
* Model architecture
* Quality and diversity of generated names

**Goal:** Apply the MLP language model to Turkish data and compare it with the simpler bigram approach.

**Amaç:** MLP language model'i Türkçe veri üzerinde çalıştırmak ve daha basit bigram modeliyle karşılaştırmak.

**File:** `07_turkish_mlp.py`

---

# ⭐ Optional Exercises / Ek Görevler

For an additional challenge, choose **one** of Karpathy's exercises.

### E01 — Zero Initialization

Initialize all weights and biases to zero.

Train the model and investigate what the model can partially learn.

Use gradients and activation values to understand what is happening.

**Video:** Part 3 — E01

---

### E02 — Fold BatchNorm into Linear

After training, fold the BatchNorm parameters into the preceding Linear layer's `W` and `b`.

Verify that the forward pass produces the same results before and after folding.

**Video:** Part 3 — E02

---

### E03 — Beat the Validation Loss

Tune the hyperparameters and try to achieve a validation loss lower than Karpathy's reported `2.2`.

Document which changes were made and how they affected the validation loss.

**Video:** Part 2 — E01

---

# 🧠 Learning Notes / Öğrenme Notları

This section will contain my own notes, observations, experiments, and conclusions from the week's materials.

Bu bölümde hafta boyunca öğrendiğim kavramları, yaptığım deneyleri, gözlemlerimi ve çıkardığım sonuçları kendi cümlelerimle paylaşacağım.

Questions to explore:

* Why are embeddings more useful than one-hot vectors?
* What does an embedding dimension represent?
* Why does using three previous characters provide more context?
* What is the trade-off of increasing the context window?
* Why do we use minibatches?
* How should a learning rate be selected?
* Why do we need train, dev, and test sets?
* What does overfitting a minibatch tell us?
* Why does `tanh` saturate?
* Why does saturation cause gradients to become small?
* How does Kaiming initialization help?
* What does BatchNorm normalize?
* Why are training and inference statistics handled differently?
* What patterns can be found in the learned embedding space?
* Why does the MLP produce different names from the bigram model?

---

# 📊 Experiments / Deneyler

The main experiments of this week will compare how different architectural and training choices affect the model.

```text
Embedding size
      ↓
Hidden layer size
      ↓
Learning rate
      ↓
Initialization
      ↓
BatchNorm
      ↓
Dev loss
```

The goal is not only to obtain a lower loss, but to understand **why** each change affects the model.

Amaç yalnızca daha düşük loss elde etmek değil, yapılan her değişikliğin modeli **neden** etkilediğini anlamak.

---

# 📁 Structure / Dosya Yapısı

```text
week-04-mlp-language-model/
│
├── README.md
├── notes.md
│
├── 01_embeddings.py
├── 02_mlp_forward.py
├── 03_training.py
├── 04_model_experiments.py
├── 05_initialization.py
├── 06_batch_norm.py
├── 07_turkish_mlp.py
│
└── optional/
    ├── e01_zero_initialization.py
    ├── e02_fold_batchnorm.py
    └── e03_hyperparameter_tuning.py
```

---

# 🎯 Outcome / Hafta Sonu Hedefi

### English

By the end of this week, I should be able to build and train an MLP-based character-level language model using learned embeddings and a multi-character context window.

I should also understand how minibatches, learning rate, dataset splits, initialization, activation saturation, and BatchNorm affect training and generalization.

Finally, I should be able to visualize learned embeddings and compare the MLP language model with the simpler bigram model.

### Türkçe

Bu haftanın sonunda öğrenilen embedding'leri ve birden fazla karakterden oluşan context window'u kullanan bir MLP tabanlı character-level language model oluşturup eğitebilmeliyim.

Ayrıca minibatch, learning rate, train/dev/test ayrımı, initialization, activation saturation ve BatchNorm'un eğitim ve genelleme üzerindeki etkilerini anlayabilmeliyim.

Son olarak öğrenilen embedding'leri görselleştirebilmeli ve MLP language model'i daha basit olan bigram modeliyle karşılaştırabilmeliyim.
