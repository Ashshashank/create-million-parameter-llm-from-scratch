# 2.3MParams-LLM-From-Scratch-Python

<a href="https://colab.research.google.com/drive/1AlnGsNU3BauFhn3ZY6tk71G9ANUAQRtx?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
</a>

<img src="https://i.ibb.co/r56NHtM/1-ox3h-To-PFUWx-Aw-URx-YEXi-Gg-removebg-preview.png" alt="Cropped Image">

Developing a custom Large Language Model (LLM) is a significant undertaking that many major technology organizations are pursuing. While many resources focus heavily on theory, this project provides a practical, code-first approach to building an LLM from the ground up.

This project is an implementation and detailed extension of the architecture explored in the repository by Brian Kitano, providing a comprehensive walkthrough of the LLaMA approach.

In this implementation, I develop an LLM with 2.3 million parameters that can be trained without the need for high-end GPU hardware. Following the LLaMA 1 Paper approach, this project uses a streamlined dataset to demonstrate the fundamental steps of creating a million-parameter model.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Understanding the Transformer Architecture of LLaMA](#understanding-the-transformer-architecture-of-llama)
  - [Pre-normalization Using RMSNorm](#pre-normalization-using-rmsnorm)
  - [SwiGLU Activation Function](#swiglu-activation-function)
  - [Rotary Embeddings (RoPE)](#rotary-embeddings-rope)
- [Setting the Stage](#setting-the-stage)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Strategy](#evaluation-strategy)
- [Setting Up a Base Neural Network Model](#setting-up-a-base-neural-network-model)
- [Replicating LLaMA Architecture](#replicating-llama-architecture)
  - [RMSNorm for pre-normalization](#rmsnorm-for-pre-normalization)
  - [Rotary Embeddings](#rotary-embeddings)
  - [SwiGLU activation function](#swiglu-activation-function)
- [Experimenting with hyperparameters](#experimenting-with-hyperparameters)
- [Saving Your Language Model (LLM)](#saving-your-language-model-llm)
- [Conclusion](#conclusion)

## Prerequisites

A basic understanding of object-oriented programming (OOP) and neural networks (NN) is required. Familiarity with PyTorch is recommended for the coding implementation.

| Topic               | Video Link                                                |
|---------------------|-----------------------------------------------------------|
| OOP                 | [OOP Video](https://www.youtube.com/watch?v=Ej_02ICOIgs) |
| Neural Network      | [Neural Network Video](https://www.youtube.com/watch?v=Jy4wM2X21u0) |
| Pytorch             | [Pytorch Video](https://www.youtube.com/watch?v=V_xro1bcAuA) |

## Understanding the Transformer Architecture of LLaMA

To build a custom LLM using the LLaMA approach, it is essential to understand its specific architecture. Below is a comparison between the vanilla transformer and the LLaMA architecture.

<img src="https://cdn-images-1.medium.com/max/25620/1*nt-ydHhSVsaLXq_HZRaLQA.png" alt="Difference between Transformers and Llama architecture" style="width: 50%;">
(Llama architecture by Umar Jamil)

### Pre-normalization Using RMSNorm:

LLaMA employs RMSNorm for normalizing the input of each transformer sub-layer. This method optimizes computational costs associated with Layer Normalization. RMSNorm provides similar performance to LayerNorm but reduces running time by approximately 7% to 64%.

<img src="https://cdn-images-1.medium.com/max/3604/1*9FA6P93WhRuWFXxVlPG3LA.png" alt="Root Mean Square Layer Normalization Paper" style="width: 50%;">

It achieves this by emphasizing re-scaling invariance and regulating inputs based on the root mean square (RMS) statistic, simplifying LayerNorm by removing the mean statistic.

### SwiGLU Activation Function:

LLaMA introduces the SwiGLU activation function. SwiGLU extends the Swish activation function and involves a custom layer with a dense network to split and multiply input activations, enhancing the expressive power of the model.

<img src="https://cdn-images-1.medium.com/max/13536/1*N3dwnqNUD0TdwPYO0NlhYg.png" alt="SwiGLU: GLU Variants Improve Transformer" style="width: 50%;">

### Rotary Embeddings (RoPE):

Rotary Embeddings (RoPE) encode absolute positional information using a rotation matrix and include explicit relative position dependency in self-attention. This offers advantages such as scalability to various sequence lengths and decaying inter-token dependency with increasing relative distances.

In addition to these concepts, the LLaMA paper introduces the use of the AdamW optimizer, efficient causal multi-head attention operators, and manually implemented backward functions to optimize computation.

Special acknowledgment to Anush Kumar for providing in-depth explanations of these LLaMA aspects.

## Setting the Stage

The project utilizes several standard Python libraries:

```python
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request
```

Configuration for model parameters:
```python
MASTER_CONFIG = {
    # Parameters added during implementation
}
```

## Data Preprocessing

While the original LLaMA was trained on 1.4 trillion tokens, this implementation uses a scaled-down approach with the TinyShakespeare dataset, containing approximately 1 million characters.

Download the dataset:
```python
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_name = "tinyshakespeare.txt"
urllib.request.urlretrieve(url, file_name)
```

Determine vocabulary size:
```python
lines = open("tinyshakespeare.txt", 'r').read()
vocab = sorted(list(set(lines)))
print('Total number of characters (Vocabulary Size):', len(vocab))
```

Create mappings:
```python
itos = {i: ch for i, ch in enumerate(vocab)}
stoi = {ch: i for i, ch in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])
```

Convert to PyTorch tensor:
```python
dataset = torch.tensor(encode(lines), dtype=torch.int8)
```

Batch generation function:
```python
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()
    return x, y

MASTER_CONFIG.update({
    'batch_size': 8,
    'context_window': 16
})
```

## Evaluation Strategy

A dedicated function for continuous evaluation during training:

```python
@torch.no_grad()
def evaluate_loss(model, config=MASTER_CONFIG):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, split, config['batch_size'], config['context_window'])
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()
    return out
```

## Setting Up a Base Neural Network Model

The base model serves as a starting point before applying LLaMA-specific optimizations.

```python
class SimpleBrokenModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    def forward(self, idx, targets=None):
        x = self.embedding(idx)
        a = self.linear(x)
        logits = F.softmax(a, dim=-1)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
        else:
            return logits
```

## Replicating LLaMA Architecture

The implementation integrates three key modifications: RMSNorm, Rotary Embeddings, and SwiGLU.

### RMSNorm for pre-normalization:

```python
class RMSNorm(nn.Module):
    def __init__(self, layer_shape, eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        ff_rms = torch.linalg.norm(x, dim=(1,2)) * x[0].numel() ** -.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[:x.shape[1], :].unsqueeze(0) * raw
```

### Rotary Embeddings:

Implementation of the position embedding rotation:

```python
def get_rotary_matrix(context_window, embedding_dim):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000. ** (-2. * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = np.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = np.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = np.cos(m_theta)
    return R
```

### SwiGLU activation function:

```python
class SwiGLU(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(self.beta * self.linear_gate(x))
        out = swish_gate * self.linear(x)
        return out
```

The final Llama model combines these components into a multi-layer architecture:

```python
class Llama(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.llama_blocks = nn.Sequential(
            OrderedDict([(f"llama_{i}", LlamaBlock(config)) for i in range(config['n_layers'])])
        )
        self.ffn = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            SwiGLU(config['d_model']),
            nn.Linear(config['d_model'], config['vocab_size']),
        )

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)
        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss
```

## Experimenting with hyperparameters

Hyperparameter tuning is essential for optimization. While the original paper used Cosine Annealing, this project allows for flexible experimentation with different schedules and optimizers.

```python
llama_optimizer = torch.optim.Adam(
    llama.parameters(),
    betas=(.9, .95),
    weight_decay=.1,
    eps=1e-9,
    lr=1e-3
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(llama_optimizer, 300, eta_min=1e-5)
```

## Saving Your Language Model (LLM)

Models can be saved in standard PyTorch format or exported for use with the Hugging Face Transformers library.

```python
# Save model parameters
torch.save(llama.state_dict(), 'llama_model_params.pth')

# Export to Transformers format
from transformers import GPT2LMHeadModel, GPT2Config
llama_config = GPT2Config.from_dict(MASTER_CONFIG)
llama_transformers = GPT2LMHeadModel(config=llama_config)
llama_transformers.load_state_dict(llama.state_dict())
llama_transformers.save_pretrained("llama_model_transformers")
```

## Conclusion

This project demonstrates the step-by-step implementation of the LLaMA architecture to build a functional, small-scale Language Model. By focusing on core components like RMSNorm, RoPE, and SwiGLU, we can create models that effectively comprehend language patterns even with limited computational resources.

## About the Developer

Shashank Adepu is an AI Engineer and Data Engineer with over 5 years of experience designing scalable machine learning and cloud data solutions across healthcare, insurance, and financial services. He specializes in Python, PyTorch, and cloud-native data platforms, with a focus on building intelligent analytics and optimizing large-scale data processing systems.

- Email: adepushashank85@gmail.com
- LinkedIn: https://www.linkedin.com/in/shashank1ad/
- GitHub: https://github.com/shashank1ad