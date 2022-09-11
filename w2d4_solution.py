r"""
# Introduction

This is a day on mechanistic interpretability of transformers. Mechanistic interpretability is the study of taking trained neural networks and reverse engineering the algorithms they've learned. It still somewhat surprises me that this is actually possible, but there are pretty strong indications that networks actually learn interpretable algorithms and that with effort and the right techniques these can be reverse engineered. There are two broad parts of this effort - finding ways to interpret the activations within a network as meaningful features (e.g. [that ResNet neurons often represent meaningful concepts](https://distill.pub/2017/feature-visualization/)), and understanding the algorithms learned by the weights to create these features (often referred to as circuits).

I (Neel) think this is a really important subfield of alignment. It seems like there are a lot of different ways that actually understanding what's going on inside models could help the development of AGI to go better for the world ([here's a list that I made](https://www.alignmentforum.org/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability)). I'm personally most excited about using them to detect deception or inner misalignment in models, which might be near impossible to detect using purely input and output based methods.

The focus today and tomorrow will be on mechanistic interpretability of transformers. Today will mostly be spent reading and understanding the Anthropic paper [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), which builds a mathematical framework for understanding the internal mechanisms of attention-only transformers (ie no MLPs) and conceptually how to break it down into (hopefully) interpretable chunks. The morning will be spent reading it **(please check my annotated table of contents below before reading!)**, and the afternoon spent going through some exercises clarifying and demonstrating the ideas in the paper, followed by your choice of interpreting small attention-only models and trying to replicate the results in the paper and looking for circuits in GPT-2.

A note: This an extremely young and pre-paradigmatic field! Mechanistic interpretability of transformers has only really been a thing for the past year and a half. There are a ton of fascinating open questions, and room to contribute to the field! Personally, it deeply offends me that there exist computer programs like GPT-3 that can basically speak English at a human level, but that we have no idea how they work and could not code up such programs ourselves.

# Reading Transformer Circuits

The goal for the morning is to:
1. Review the below notes and table of contents for the paper before reading it
2. Skim the reading tips below, making a mental note to refer back as needed
3. Read and understand the paper: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)

There are a bunch of subtle but key ideas in here, and I recommend regularly stopping to practice explaining bits of the paper to your partner and vice versa, and seeing if you're on the same page. I (Neel) was an author on the paper and will be hanging out in the main room all morning to answer questions - feel free to read in there or come over and ask me things.

The following is an annotated table of contents to sections of the paper. **Bold** means the section is important, normal means it's worth reading but not essential, and *italics* means skip. **I recommend prioritising the bold sections and what feels interesting to you over fully reading through in order** (though you may need to backtrack in case you skip key definitions or details). The table of contents is followed by a list of tips for understanding the content, which it's worth skimming over before reading and coming back to when you reach the relevant section (though some won't make sense until you've read the paper, so don't spend too much time on this).

## Annotated Table of Contents of the Paper

* **[Introduction](https://transformer-circuits.pub/2021/framework/index.html)**
* **[Summary of Results](https://transformer-circuits.pub/2021/framework/index.html#summary-of-results)**
* **[Transformer Overview](https://transformer-circuits.pub/2021/framework/index.html#transformer-overview)** **(Most important section!)**
* [Zero-Layer Transformers](https://transformer-circuits.pub/2021/framework/index.html#zero-layer-transformers)
* [One-Layer Attention-Only Transformers](https://transformer-circuits.pub/2021/framework/index.html#one-layer-attention-only-transformers)
    * [The Path Expansion Trick](https://transformer-circuits.pub/2021/framework/index.html#1l-path-expansion)
    * **[Splitting Attention Head terms into Query-Key and Output-Value Circuits](https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits)**
    * **[Interpretation as Skip-Trigrams](https://transformer-circuits.pub/2021/framework/index.html#interpretation-as-skip-trigrams)**
    * [*Summarizing OV/QK Matrices*](https://transformer-circuits.pub/2021/framework/index.html#summarizing-ovqk-matrices) (Skip, somewhat convoluted and mathsy and doesn't really generalise)
    * [Do We "Fully Understand" One-Layer Models?](https://transformer-circuits.pub/2021/framework/index.html#do-we-fully-understand-one-layer-models)
* [Two-Layer Attention-Only Transformers](https://transformer-circuits.pub/2021/framework/index.html#two-layer-attention-only-transformers)
    * **[Three Kinds of Composition](https://transformer-circuits.pub/2021/framework/index.html#three-kinds-of-composition)**
    * **[Path Expansion of Logits](https://transformer-circuits.pub/2021/framework/index.html#path-expansion-of-logits)**
    * [*Path Expansion of Attention Scores QK Circuit*](https://transformer-circuits.pub/2021/framework/index.html#path-expansion-of-attention-scores-qk-circuit) (Skip, v convoluted & mathsy)
    * **[Analyzing a Two-Layer Model](https://transformer-circuits.pub/2021/framework/index.html#analyzing-a-two-layer-model)**
    * **[Induction Heads](https://transformer-circuits.pub/2021/framework/index.html#induction-heads)**
    * [Term Importance Analysis](https://transformer-circuits.pub/2021/framework/index.html#term-importance-analysis)
    * [Virtual Attention Heads](https://transformer-circuits.pub/2021/framework/index.html#virtual-attention-heads)
* [Where Does This Leave Us?](https://transformer-circuits.pub/2021/framework/index.html#where-does-this-leave-us)
* [*Related Work*](https://transformer-circuits.pub/2021/framework/index.html#related-work)
* [Additional Intuitions](https://transformer-circuits.pub/2021/framework/index.html#additional-intuition)
    * **[MLP Layers](https://transformer-circuits.pub/2021/framework/index.html#additional-intuition:~:text=Intuition%20and%20Observations-,MLP%20Layers,-This%20article%20has)**

## Tips & Insights for Reading the Paper
* The eigenvalue stuff is very cool, but doesn't generalise that much, it's not a priority to get your head around
* It's really useful to keep clear in your head the difference between parameters (learned numbers that are intrinsic to the network and independent of the inputs) and activations (temporary numbers calculated during a forward pass, that are functions of the input).
    * Attention is a slightly weird thing - it's an activation, but is also used in a matrix multiplication with another activation (z), which makes it parameter-y.
        * The idea of freezing attention patterns disentangles this, and lets us treat it as parameters.
* The residual stream is the fundamental object in a transformer - each layer just applies incremental updates to it - this is really useful to keep in mind throughout!
    * This is in contrast to a classic neural network, where each layer's output is the central object
    * To underscore this, a funky result about transformers is that the aspect ratio isn't *that* important - if you increase d_model/n_layer by a factor of 10 from optimal for a 1.5B transformer (ie controlling for the number of parameters), then loss decreases by <1%.
* The calculation of attention is a bilinear form (ie via the QK circuit) - for any pair of positions it takes an input vector from each and returns a scalar (so a ctx x ctx tensor for the entire sequence), while the calculation of the output of a head pre weighting by attention (ie via the OV circuit) is a linear map from the residual stream in to the residual stream out - the weights have the same shape, but are doing functions of completely different type signatures!
* How to think about attention: A framing I find surprisingly useful is that attention is the "wiring" of the neural network. If we hold the attention patterns fixed, they tell the model how to move information from place to place, and thus help it be effective at sequence prediction. But the key interesting thing about a transformer is that attention is *not* fixed - attention is computed and takes a substantial fraction of the network's parameters, allowing it to dynamically set the wiring. This can do pretty meaningful computation, as we see with induction heads, but is in some ways pretty limited. In particular, if the wiring is fixed, an attention only transformer is a purely linear map! Without the ability to intelligently compute attention, an attention-only transformer would be incredibly limited, and even with it it's highly limited in the functional forms it can represent.
    * Another angle - attention as generalised convolution. A naive transformer would use 1D convolutions on the sequence. This is basically just attention patterns that are hard coded to be uniform over the last few tokens - since information is often local, this is a decent enough default wiring. Attention allows the model to devote some parameters to compute more intelligent wiring, and thus for a big enough and good enough model will significantly outperform convolutions.
* One of the key insights of the framework is that there are only a few activations of the network that are intrinsically meaningful and interpretable - the input tokens, the output logits and attention patterns (and neuron activations in non-attention-only models). Everything else (the residual stream, queries, keys, values, etc) are just intermediate states on a calculation between two intrinsically meaningful things, and you should instead try to understand the start and the end. Our main goal is to decompose the network into many paths between interpretable start and end states
    * We can get away with this because transformers are really linear! The composition of many linear components is just one enormous matrix
* A really key thing to grok about attention heads is that the QK and OV circuits act semi-independently. The QK circuit determines which previous tokens to attend to, and the OV circuit determines what to do to tokens *if* they are attended to. In particular, the residual stream at the destination token *only* determines the query and thus what tokens to attend to - what the head does *if* it attends to a position is independent of the destination token residual stream (other than being scaled by the attention pattern).
    <p align="center">
        <img src="w2d4_Attn_Head_Pic.png" width="400" />
    </p>
* Skip trigram bugs are a great illustration of this - it's worth making sure you really understand them. The key idea is that the destination token can *only* choose what tokens to pay attention to, and otherwise not mediate what happens *if* they are attended to. So if multiple destination tokens want to attend to the same source token but do different things, this is impossible - the ability to choose the attention pattern is insufficient to mediate this.
    * Eg, keep...in -> mind is a legit skip trigram, as is keep...at -> bay, but keep...in -> bay is an inherent bug from this pair of skip trigrams
* The tensor product notation looks a lot more complicated than it is. $A \otimes W$ is shorthand for "the function $f_{A,W}$ st $f_{A,W}(x)=AxW$" - I recommend mentally substituting this in in your head everytime you read it.
* K, Q and V composition are really important and fairly different concepts! I think of each attention head as a circuit component with 3 input wires (Q,K,V) and a single output wire (O). Composition looks like connecting up wires, but each possible connection is a choice! The key, query and value do different things and so composition does pretty different things.
    * Q-Composition, intuitively, says that we want to be more intelligent in choosing our destination token - this looks like us wanting to move information to a token based on *that* token's context. A natural example might be the final token in a several token word or phrase, where earlier tokens are needed to disambiguate it, eg E|iff|el| Tower|
    * K-composition, intuitively, says that we want to be more intelligent in choosing our source token - this looks like us moving information *from* a token based on its context (or otherwise some computation at that token).
        * Induction heads are a clear example of this - the source token only matters because of what comes before it!
    * V-Composition, intuitively, says that we want to *route* information from an earlier source token *other than that token's value* via the current destination token. It's less obvious to me when this is relevant, but could imagine eg a network wanting to move information through several different places and collate and process it along the way
        * One example: In the ROME paper, we see that when models recall that "The Eiffel Tower is in" -> " Paris", it stores knowledge about the Eiffel Tower on the " Tower" token. When that information is routed to | in|, it must then map to the output logit for | Paris|, which seems likely due to V-Composition
* A surprisingly unintuitive concept is the notion of heads (or other layers) reading and writing from the residual stream. These operations are *not* inverses! A better phrasing might be projecting vs embedding.
    * Reading takes a vector from a high-dimensional space and *projects* it to a smaller one - (almost) any two pair of random vectors will have non-zero dot product, and so every read operation can pick up *somewhat* on everything in the residual stream. But the more a vector is aligned to the read subspace, the most that vector's norm (and intuitively, its information) is preserved, while other things are lower fidelity
        * A common reaction to these questions is to start reasoning about null spaces, but I think this is misleading - rank and nullity are discrete concepts, while neural networks are fuzzy, continuous objects - nothing ever actually lies in the null space or has non-full rank (unless it's explicitly factored). I recommend thinking in terms of "what fraction of information is lost". The null space is the special region with fraction lost = 1
    * Writing *embeds* a vector into a small dimensional subspace of a larger vector space. The overall residual stream is the sum of many vectors from many different small subspaces.
        * Every read operation can see into every writing subspace, but will see some with higher fidelity, while others are noise it would rather ignore.
    * It can be useful to reason about this by imagining that d_head=1, and that every vector is a random Gaussian vector - projecting a random Gaussian onto another in $\mathbb{R}^n$ will preserve $\frac{1}{n}$ of the variance, on average.
* A key framing of transformers (and neural networks in general) is that they engage in **lossy compression** - they have a limited number of dimensions and want to fit in more directions than they have dimensions. Each extra dimension introduces some interference, but has the benefit of having more expressibility. Neural networks will learn an optimal-ish solution, and so will push the compression as far as it can until the costs of interference dominate.
    * This is clearest in the case of QK and OV circuits - $W_QK=W_Q^TW_K$ is a d_model x d_model matrix with rank d_head. And to understand the attention circuit, it's normally best to understand $W_QK$ on its own. Often, the right mental move is to forget that $W_QK$ is low rank, to understand what the ideal matrix to learn here would be, and then to assume that the model learns the best low rank factorisation of that.
        * This is another reason to not try to interpret the keys and queries - the intermediate state of a low rank factorisations are often a bit of a mess because everything is so compressed (though if you can do SVD on $W_QK$ that may get you a meaningful basis?)
        * Rough heuristic for thinking about low rank factorisations and how good they can get - a good way to produce one is to take the SVD and zero out all but the first d_head singular values.
    * This is the key insight behind why polysemanticity (back from w1d5) is a thing and is a big deal - naturally the network would want to learn one feature per neuron, but it in fact can learn to compress more features than total neurons. It has some error introduced from interference, but this is likely worth the cost of more compression.
        * Just as we saw there, the sparsity of features is a big deal for the model deciding to compress things! Inteference cost goes down the more features are sparse (because unrelated features are unlikely to co-occur) while expressibility benefits don't really change that much.
    * The residual stream is the central example of this - every time two parts of the network compose, they will be communicating intermediate states via the residual stream. Bandwidth is limited, so these will likely try to each be low rank. And the directions within that intermediate product will *only* make sense in the context of what the writing and reading components care about. So interpreting the residual stream seems likely fucked - it's just
* The 'the residual stream is fundamentally uninterpretable' claim is somewhat overblown - most models do dropout on the residual stream which somewhat privileges that basis
    * And there are [*weird*](https://timdettmers.com/2022/08/17/llm-int8-and-emergent-features/) results about funky directions in the residual stream.
* Getting your head around the idea of a privileged basis is very worthwhile! The key mental move is to flip between "a vector is a direction in a geometric space" and "a vector is a series of numbers in some meaningful basis, where each number is intrinsically meaningful". By default, it's easy to spend too much time in the second mode, because every vector is represented as a series of numbers within the GPU, but this is often less helpful!

<details>
<summary>An aside on why we need the tensor product notation at all</summary>

Neural networks are functions, and are built up of several subcomponents (like attention heads) that are also functions - they are defined by how they take in an input and return an output. But when doing interpretability we want the ability to talk about the network as a function intrinsically and analyse the structure of this function, *not* in the context of taking in a specific input and getting a specific output. And this means we need a language that allows us to naturally talk about functions that are the sum (components acting in parallel) or composition (components acting in series) of other functions.

A simple case of this: We're analysing a network with several linear components acting in parallel - component $C_i$ is the function $x \rightarrow W_ix$, and can be represented intrinsically as $W_i$ (matrices are equivalent to linear maps). We can represent the layer with all acting in parallel as $x \rightarrow \sum_i W_ix=(\sum_i W_i)x$, and so intrinsically as $\sum_i W_i$ - this is easy because matrix notation is designed to make addition of.

Attention heads are harder because they map the input tensor $x$ (shape: `[position x d_model]`) to an output $Ax(W_OW_V)^T$ - this is a linear function, but now on a *tensor*, so we can't trivially represent addition and composition with matrix notation. The paper uses the notation $A\otimes W_OW_V$, but this is just different notation for the same underlying function. The output of the layer is the sum over the 12 heads: $\sum_i A^{(i)}x(W_O^{(i)}W_V^{(i)})^T$. And so we could represent the function of the entire layer as $\sum_i A^{(i)} x (W_O^{(i)}W_V^{(i)})$. There are natural extensions of this notation for composition, etc, though things get much more complicated when reasoning about attention patterns - this is now a bilinear function of a pair of inputs: the query and key residual streams. (Note that $A$ is also a function of $x$ here, in a way that isn't obvious from the notation.)

The key point to remember is that if you ever get confused about what a tensor product means, explicitly represent it as a function of some input and see if things feel clearer.
</details>

# Rest of the Day


We're going to spend the rest of the day analysing our very own 2L Attn-Only model. We're going to first validate some of the paper's claims about ways to understand attention heads, then build some rudimentary interpretability tooling, and finally use this tooling to reverse engineer the induction heads learned by the model.

**Meta:** The main goal of the above structure is to help you really understand the ideas in the paper, and to practice careful and rigorous reverse engineering of a circuit. But more generally, the main thing I want to convey today is that interpreting models is tractable, and that they can actually be understood! (And hopefully, that this is exciting and fascinating!) If that structure doesn't sound exciting to you, feel free to skip around, and I give some tips and infrastructure at the end for looking for circuits in GPT-2 - feel free to just skip to that if you ever get bored! If you do skip ahead, I recommend copying code from the solutions file, as some of the functions we define here may be pretty useful.

**Meta 2:** I leave a lot of random asides and exercises in here, based on concepts I think are cute or useful. I think it'd take more time than a day to go through them all, and recommend skipping/having a low bar for looking up solutions/following your curiosity.

## Table of Contents

<!-- TOC -->

- [Introduction](#introduction)
- [Reading Transformer Circuits](#reading-transformer-circuits)
    - [Annotated Table of Contents of the Paper](#annotated-table-of-contents-of-the-paper)
    - [Tips & Insights for Reading the Paper](#tips--insights-for-reading-the-paper)
- [Rest of the Day](#rest-of-the-day)
    - [Table of Contents](#table-of-contents)
- [Introducing Our Toy Attention-Only Model](#introducing-our-toy-attention-only-model)
- [Hook Points](#hook-points)
- [Understanding Attention Heads](#understanding-attention-heads)
    - [QK-Circuits](#qk-circuits)
- [Building Interpretability Tools](#building-interpretability-tools)
    - [Direct Logit attribution](#direct-logit-attribution)
    - [Visualising Attention Patterns](#visualising-attention-patterns)
    - [Summarising attention patterns](#summarising-attention-patterns)
    - [Ablations](#ablations)
    - [Finding Induction Circuits](#finding-induction-circuits)
    - [Checking for the induction capability](#checking-for-the-induction-capability)
    - [Looking for Induction Attention Patterns](#looking-for-induction-attention-patterns)
        - [Logit Attribution](#logit-attribution)
    - [Ablations](#ablations)
    - [Previous Token Head](#previous-token-head)
    - [Mechanistic Analysis of Induction Circuits](#mechanistic-analysis-of-induction-circuits)
        - [Reverse Engineering OV-Circuit Analysis](#reverse-engineering-ov-circuit-analysis)
        - [Reverse Engineering Positional Embeddings + Prev Token Head](#reverse-engineering-positional-embeddings--prev-token-head)
        - [Composition Analysis](#composition-analysis)
            - [Splitting activations](#splitting-activations)
            - [Interpreting the K-Composition Circuit](#interpreting-the-k-composition-circuit)
    - [Further Exploration of Induction Circuits](#further-exploration-of-induction-circuits)
        - [Composition scores](#composition-scores)
            - [Setting a Baseline](#setting-a-baseline)
            - [Theory + Efficient Implementation](#theory--efficient-implementation)
            - [Targeted Ablations](#targeted-ablations)
- [Bonus](#bonus)
    - [Looking for Circuits in Real LLMs](#looking-for-circuits-in-real-llms)
    - [Training Your Own Toy Models](#training-your-own-toy-models)
    - [Interpreting Induction Heads During Training](#interpreting-induction-heads-during-training)

<!-- /TOC -->


# Introducing Our Toy Attention-Only Model

Here we introduce a toy 2L attention-only transformer trained specifically for today. Some changes to make them easier to interpret:
- It has only attention blocks
- The positional embeddings are only added to each key and query vector in the attention layers as opposed to the token embeddings (meaning that the residual stream can't directly encode positional information)
  - This turns out to make it *way* easier for induction heads to form, it happens 2-3x times earlier - [see the comparison of two training runs](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-83---VmlldzoyNTI0MDMz?accessToken=8ap8ir6y072uqa4f9uinotdtrwmoa8d8k2je4ec0lyasf1jcm3mtdh37ouijgdbm) here. (The bump in each curve is the formation of induction heads)
- It has no MLP layers, no LayerNorms, and no biases
- There are separate embed and unembed matrices (ie the weights are not tied)
- The activations in the attention layers $(q, k, v, z)$ have shape `[batch, position, head_index, d_head]` (ie, not flattened into a single d_model axis)
  - Similarly $W_K, W_Q, W_V$ have shape `[head_index, d_head, d_model]`, $W_O$ has shape `[head_index, d_model, d_head]`
- Convention: All weight matrices multiply on the left (ie have shape `[output, input]`)

See the model code in `w2d4_attn_only_transformer.py`

Here are its specs:
- `num_layers`: 2,
- `num_heads`: 12,
- `head_size`: 64, (`d_head` in Anthropic notation)
- `hidden_size`: 768, (`d_model` in Anthropic notation)
- `vocab_size`: 50259,
- `max_position_embeddings`: 2048
- GPT-NeoX tokenizer
  - The tokenizer was adapted to add a special token to the start of each string, <|endoftext|>. The method `model.to_tokens(text)` automatically tokenizes with this.
- Trained on the Pile, weight decay 0.01

Finally,  the pseudocode for the model is shown below. Note that activation and weight names and shapes differ from GPT-2. See `w2d4_attn_only_transformer.py` for the full implementation.

```python
def Transformer(tokens):
    embed[batch, position, d_model] = Embed(tokens[batch, position])
    pos_embed[batch, position, d_model] = PosEmbed(tokens[batch, position])
    residual[batch, position, d_model] = embed[batch, position, d_model]
    residual[batch, position, d_model] = AttnBlock0(residual[batch, position, d_model])
    residual[batch, position, d_model] = AttnBlock1(residual[batch, position, d_model])
    return logits[batch, position, d_vocab] = Unembed(residual[batch, position, d_model])

def Embed(tokens[batch, position]):
    return embed[batch, position, d_model] = W_E[d_model, tokens[batch, position]]

def PosEmbed(tokens[batch, position]):
    return pos_embed[batch, position, d_model] = W_pos[d_model, position]

def Unembed(residual[batch, position, d_model]):
    return logits[batch, position, d_vocab] = W_U[d_vocab, **d_model**] @ residual[batch, position, **d_model**]

def AttnBlock(residual[batch, position, d_model], pos_embed[batch, position, d_model]):
    resid_pre[batch, position, d_model] = residual[batch, position, d_model]
    attn_out[batch, position, d_model] = Attn(resid_pre[batch, position, d_model], pos_embed[batch, position, d_model])
    return resid_post[batch, position, d_model] = resid_pre[batch, position, d_model] + attn_out[batch, position, d_model]

def Attn(residual[batch, position, d_model], pos_embed[batch, position, d_model]):
    qk_input[batch, position, d_model] = residual[batch, position, d_model] + pos_embed[batch, position, d_model]
    k[batch, position, head_index, d_head] = W_K[head_index, d_head, **d_model**] @ qk_input[batch, position, **d_model**]
    q[batch, position, head_index, d_head] = W_Q[head_index, d_head, **d_model**] @ qk_input[batch, position, **d_model**]
    v[batch, position, head_index, d_head] = W_V[head_index, d_head, **d_model**] @ residual[batch, position, **d_model**]

    attn_scores[batch, head_index, query_pos, key_pos] = causal_mask(q[batch, query_pos, head_index, **d_head**] @ k[batch, key_pos, head_index, **d_head**])
    attn_pattern[batch, head_index, query_pos, key_pos] = softmax(attn_scores[batch, head_index, query_pos, key_pos], dim=key_pos)

    z[batch, query_pos, head_index, d_head] = v[batch, **key_pos**, head_index, d_head] @ attn_pattern[batch, head_index, query_pos, **key_pos**]
    result[batch, query_pos, head_index, d_model] = W_O[head_index, d_model, **d_head**] @ z[batch, query_pos, head_index, **d_head**]

    return attn_out = result[batch, query_pos, **head_index**, d_model].sum(head_index)
```


"""

# %%

import os
import sys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import w2d4_test
from w2d4_attn_only_transformer import AttnOnlyTransformer

pio.renderers.default = "notebook"

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)
# %%
"""
NOTE: If you get an error below related to not being able to load the tokenizer, please upgrade your `tranformers` version (you can do this by running `!pip install transformers --upgrade` in your interactive terminal.
"""
cfg = {
    "d_model": 768,
    "d_head": 64,
    "n_heads": 12,
    "n_layers": 2,
    "n_ctx": 2048,
    "d_vocab": 50278,
    "lr": 1e-3,
    "betas": (0.9, 0.95),
    "weight_decay": 0.01,
    "batch_size": 144,
    "batches_per_step": 2,
    "seed": 398,
    "dataset_name": "the_pile",
    "use_attn_result": True,  # This toggles whether to explicitly calculate the result of each head, the term that directly adds to the residual stream - this allows us to cache it, but takes up a lot more GPU memory. Exercise: Why?
}

if MAIN:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = AttnOnlyTransformer(cfg, tokenizer)
    pretrained_weights = t.load("w2d4_attn_only_2L.pth")
    model.load_state_dict(pretrained_weights)
    model.cuda()

# %%
"""
# Hook Points

A key thing for doing mechanistic interpretability is being able to access intermediate activations in the model, and potentially to intervene on them (eg to freeze an attention pattern). Since a forward pass in a typical PyTorch model implementation only outputs the final activations/logits, this isn't a trivial task. To make things easier, each activation in this educational transformer comes with a hook point allowing you to add a function to that activation, to either access and process it or to intervene and edit it.
> In the code, this looks like wrapping each activation in an identity function, eg `embed=self.hook_embed(embed)`

For now, the only thing we'll need is the `cache_all` command to cache all activations - this populates a dictionary mapping each activation's name to its tensor - see the pseudocode above for each activation's name and shape (note that `blocks.0.hook_resid_post == blocks.1.hook_resid_pre` - these are given for convenience)

Running the cell below will print the name and shape of each activation:
"""

if MAIN:
    cache_example = {}
    model.cache_all(cache_example)
    example_text = "IT WAS A BRIGHT cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
    # Use model.to_tokens rather than the tokenizer - the model was trained with an <endoftext> token at the start of the context, and this method will add that.
    example_tokens = model.to_tokens(example_text)
    print(f"There are {example_tokens.shape[-1]} tokens")
    logits = model(example_tokens)
    # This line turns off caching but does not delete the cache. This means future runs of the model won't overwrite the cache
    model.reset_hooks()

    for activation_name in cache_example:
        activation = cache_example[activation_name]
        print(f"Activation: {activation_name} Shape: {activation.shape}")

# %%
"""
# Understanding Attention Heads

A bunch of the important ideas in the paper are centered on understanding an attention head as a (surprisingly linear!) mathematical object, and seeing that there are a bunch of equivalent ways to write out the function. It's worth ensuring you really get what's going on here, so let's practice a couple of equivalent ways of implementing an attention head.

## QK-Circuits

QK-circuit is just a factored matrix - queries and keys are not intrinsically interpretable and neither are $W_Q$ or $W_K$.

Implement function `QK_attn()` below which takes in the QK matrix for a single head, the attention input (ie residual stream + positional embeddings, not including the batch dimension), and returns a (softmaxed) attention pattern.

<details>

<summary>Hint 1</summary>

Remember to scale the raw attention scores by the square root of d_head, which can be found in the `cfg` dict.

</details>

<details>

<summary>Hint 2</summary>

Make sure to apply the attention mask so query tokens can't access key tokens from higher positions.

</details>

"""


def mask_scores(attn_scores):
    """Mask the attention scores so that tokens don't attend to previous tokens."""
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1e4).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores


def QK_attn(W_QK, qk_input):  # [query_d_model, key_d_model]  # [position, d_model]
    """
    W_QK: (query_d_model, key_d_model)
    qk_input: (position, d_model)
    """
    if "SOLUTION":
        attn_scores = t.einsum("qd,dD,Dk->qk", qk_input, W_QK, qk_input.T) / np.sqrt(
            cfg["d_head"]
        )
        masked_attn_scores = mask_scores(attn_scores)
        attn_pattern = F.softmax(masked_attn_scores, dim=1)
        return attn_pattern


def run_QK_attn():
    layer = 0
    head_index = 0
    batch_index = 0
    W_Q = model.blocks[layer].attn.W_Q[head_index]
    W_K = model.blocks[layer].attn.W_K[head_index]
    qk_input = (
        cache_example[f"blocks.{layer}.hook_resid_pre"][batch_index]
        + cache_example["hook_pos_embed"]
    )
    original_attn_pattern = cache_example[f"blocks.{layer}.attn.hook_attn"][
        batch_index, head_index, :, :
    ]
    W_QK = W_Q.T @ W_K
    return (QK_attn, W_QK, qk_input, original_attn_pattern)


if MAIN:
    w2d4_test.test_qk_attn(*run_QK_attn())


# %%
"""
The OV-circuit is just a factored matrix, AND attention and OV commute - we can apply attention before V and O, after V and before O, or after both, and get the same result.

Attention before V and O:
"""


def OV_result_mix_before(W_OV, residual_stream_pre, attn_pattern):
    """
    Apply attention to the residual stream, and THEN apply W_OV.
    Inputs:
        W_OV: (d_model, d_model)
        residual_stream_pre: (position, d_model)
        attn_pattern: (query_pos, key_pos)
    Returns:
        head output of shape: (position, d_model)
    """
    "SOLUTION"
    mixed_residual_stream = t.einsum("km,qk->qm", residual_stream_pre, attn_pattern)
    head_results = mixed_residual_stream @ W_OV.T
    return head_results


def run_OV_result_mix_before():
    layer = 0
    head_index = 0
    batch_index = 0
    W_O = model.blocks[layer].attn.W_O[head_index].detach().clone()
    W_V = model.blocks[layer].attn.W_V[head_index].detach().clone()
    W_OV = W_O @ W_V
    residual_stream_pre = (
        cache_example[f"blocks.{layer}.hook_resid_pre"][batch_index].detach().clone()
    )
    original_head_results = (
        cache_example[f"blocks.{layer}.attn.hook_result"][batch_index, :, head_index]
        .detach()
        .clone()
    )
    attn_pattern = (
        cache_example[f"blocks.{layer}.attn.hook_attn"][batch_index, head_index, :, :]
        .detach()
        .clone()
    )
    return (
        OV_result_mix_before,
        W_OV,
        residual_stream_pre,
        attn_pattern,
        original_head_results,
    )


if MAIN:
    (
        OV_result_mix_before,
        W_OV,
        residual_stream_pre,
        attn_pattern,
        original_head_results,
    ) = run_OV_result_mix_before()
    w2d4_test.test_ov_result_mix_before(
        OV_result_mix_before,
        W_OV,
        residual_stream_pre,
        attn_pattern,
        original_head_results,
    )

# %%
"""
Attention after V and O:
"""


def OV_result_mix_after(
    W_OV,
    residual_stream_pre,
    attn_pattern,  # [d_model, d_model]  # [batch, position, d_model]
):  # [batch, query_pos, key_pos]
    # Apply OV THEN attention
    """
    Apply W_OV to the residual stream, and THEN apply attention.
    Inputs:
        W_OV: (d_model, d_model)
        residual_stream_pre: (position, d_model)
        attn_pattern: (query_pos, key_pos)
    Returns:
        head output of shape: (position, d_model)
    """
    "SOLUTION"
    unmixed_results = residual_stream_pre @ W_OV.T
    result = t.einsum("km,qk->qm", unmixed_results, attn_pattern)
    return result


if MAIN:
    w2d4_test.test_ov_result_mix_after(
        OV_result_mix_after,
        W_OV,
        residual_stream_pre,
        attn_pattern,
        original_head_results,
    )


# %%
"""
# Building Interpretability Tools

In this next section, we're going to build some basic interpretability tools to decompose models and answer some questions about them.

Let's re-run our model on some new text - feel free to write your own!
"""
cache_2 = {}
if MAIN:
    text_2 = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    tokens_2 = model.to_tokens(text_2)
    tokens_2 = tokens_2.cuda()
    model.cache_all(cache_2)
    logits_2 = model(tokens_2)
    # This line turns off cacheing, so future runs of the model won't overwrite the cache
    model.reset_hooks()

# %%
"""
## Direct Logit attribution

A consequence of the residual stream is that the output logits are the sum of the contributions of each layer, and thus the sum of the results of each head. This means we can decompose the output logits into a term coming from each head and directly do attribution like this! Write a function to look at how much each head and the direct path term contributes to the correct logit.

<details><summary>A concrete example</summary>

Let's say that our model knows that the token Harry is followed by the token Potter, and we want to figure out how it does this. The logits on Harry are `W_U @ residual`. But this is a linear map, and the residual stream is the sum of all previous layers `residual = embed + attn_out_0 + attn_out_1`. So `logits = (W_U @ embed) + (W_U @ attn_out_0) + (W_U @ attn_out_1)`

We can be even more specific, and *just* look at the logit of the Potter token - this corresponds to a row of W_U, and so a direction in the residual stream - our logit is now a single number that is the sum of `(potter_U @ embed) + (potter_U @ attn_out_0) + (potter_U @ attn_out_1)`. Even better, we can decompose each attention layer output into the sum of the result of each head, and use this to get many terms.
</details>

Calculate the logit attributions of the following paths to the logits: direct path (via the residual connections from the embedding to unembedding); each layer 0 head (via the residual connection and skipping layer 1); each layer 1 head. To emphasise, these are not paths from the start to the end of the model, these are paths from the output of some component directly to the logits - we make no assumptions about how each path was calculated!

Note: Here we are just looking at the DIRECT effect on the logits - if heads compose with other heads and affect logits like that, or inhibit logits for other tokens to boost the correct one we will not pick up on this!

Note 2: By looking at just the logits corresponding to the correct token, our data is much lower dimensional because we can ignore all other tokens other than the correct next one (Dealing with a 50K vocab size is a pain!). But this comes at the cost of missing out on more subtle effects, like a head suppressing other plausible logits, to increase the log prob of the correct one.

Note 3: When calculating correct output logits, we will get tensors with a dimension (position - 1,), not (position,) - we remove the final element of the output (logits), and the first element of labels (tokens). This is because we're predicting the *next* token, and we don't know the token after the final token, so we ignore it.

<details><summary>Aside:</summary>

While we won't worry about this for this exercise, logit attribution is often more meaningful if we first center W_U - ie, ensure the mean of each row writing to the output logits is zero. Log softmax is invariant when we add a constant to all the logits, so we want to control for a head that just increases all logits by the same amount. We won't do this here for ease of testing.</details>

<details><summary>Exercise: Why don't we do this to the log probs instead?</summary>

A: Because log probs aren't linear, they go through log_softmax, a non-linear function.</details>
"""

# Some helper functions for nice plotting
def to_numpy(tensor):
    """Helper function to convert things to numpy before plotting with Plotly."""
    return tensor.detach().cpu().numpy()


def convert_tokens_to_string(tokens, batch_index=0):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{tokenizer.decode(tok)}|_{c}" for c, tok in enumerate(tokens)]


def plot_logit_attribution(logit_attr, tokens):
    # Remove dummy batch dimension
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [
        f"L{l}H{h}" for l in range(cfg["n_layers"]) for h in range(cfg["n_heads"])
    ]
    px.imshow(
        to_numpy(logit_attr),
        x=x_labels,
        y=y_labels,
        labels={"x": "Term", "y": "Position", "color": "logit"},
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
    ).show()


def logit_attribution(
    embed,
    l1_results,
    l2_results,
    W_U,
    tokens,
):
    """
    We have provided 'W_U_to_logits' which is a (position, d_model) tensor where each row is the unembed for the correct NEXT token at the current position.

    Inputs:
        embed: (position, d_model)
        l1_results: (position, head_index, d_model)
        l2_results: (position, head_index, d_model)
        W_U: (d_vocab, d_model)
    Returns:
        Tensor representing the concatenation (along dim=-1) of logit attributions from the direct path (position-1,1), layer 0 logits (position-1, n_heads) and layer 1 logits (position-1, n_heads).
    """
    W_U_to_logits = W_U[tokens[1:], :]
    if "SOLUTION":
        direct_path_logits = t.einsum("pm,pm->p", W_U_to_logits, embed[:-1, :])
        l1_logits = t.einsum("pm,pim->pi", W_U_to_logits, l1_results[:-1])
        l2_logits = t.einsum("pm,pim->pi", W_U_to_logits, l2_results[:-1])
        logit_attribution = t.concat(
            [direct_path_logits[:, None], l1_logits, l2_logits], dim=-1
        )
        return logit_attribution


if MAIN:
    w2d4_test.test_logit_attribution(
        logit_attribution, model, cache_2, tokens_2, logits_2
    )

# %%
"""
Now we can visualise the logit attributions for each path through the model.
"""
# %%
if MAIN:
    batch_index = 0
    embed = cache_2["hook_embed"][batch_index]
    l1_results = cache_2["blocks.0.attn.hook_result"][batch_index]
    l2_results = cache_2["blocks.1.attn.hook_result"][batch_index]
    logit_attr = logit_attribution(
        embed, l1_results, l2_results, model.unembed.W_U, tokens_2[batch_index]
    )
    plot_logit_attribution(logit_attr, tokens_2)

# %%
"""
## Visualising Attention Patterns

A key insight from the paper is that we should focus on interpreting the parts of the model that are intrinsically interpretable - the input tokens, the output logits and the attention patterns. Everything else (the residual stream, keys, queries, values, etc) are compressed intermediate states when calculating meaningful things. So a natural place to start is classifying heads by their attention patterns on various texts.

When doing interpretability, it's always good to begin by visualising your data, rather than taking summary statistics. Summary statistics can be super misleading! But now that we have visualised the attention patterns, we can create some basic summary statistics and use our visualisations to validate them! (Accordingly, being good at web dev/data visualisation is a surprisingly useful skillset! Neural networks are very high-dimensional object)

A good place to start is visualising the attention patterns of the model on input text. Go through a few of these, and get a sense for what different heads are doing.

The "animation frame" corresponds to each head in the layer. You will probably want to zoom in and out to get a better view of the activations.
"""


def plot_attn_pattern(patterns, tokens, title=None):
    # Patterns has shape [head_index, query_pos, key_pos] or [query_pos, key_pos]
    tokens_to_string = convert_tokens_to_string(tokens)
    if len(patterns.shape) == 3:
        px.imshow(
            to_numpy(patterns),
            animation_frame=0,
            y=tokens_to_string,
            x=tokens_to_string,
            labels={"x": "Key", "y": "Query"},
            color_continuous_scale="Blues",
            title=title,
        ).show()
    else:
        px.imshow(
            to_numpy(patterns),
            y=tokens_to_string,
            x=tokens_to_string,
            labels={"x": "Key", "y": "Query"},
            color_continuous_scale="Blues",
            title=title,
        ).show()


if MAIN:
    for layer in range(cfg["n_layers"]):
        plot_attn_pattern(
            cache_2[f"blocks.{layer}.attn.hook_attn"][0],
            tokens_2,
            f"Layer {layer} attention patterns",
        )


# %%
"""
## Summarising attention patterns
Three basic patterns for attention heads are those that mostly attend to the current token, the previous token, or the first token (often used as a resting or null position for heads that only sometimes activate). Let's make detectors for those! Validate your detectors by comparing these results to the visual attention patterns above - summary statistics on their own can be dodgy, but are much more reliable if you can validate it by directly playing with the data.

<details>

<summary>Hint</summary>

Use `t.diagonal` on the last two diagonals and vary the offset for the current and prev token detectors.

</details>

Bonus: Try inputting different text, and see how stable your results are.
"""


def current_attn_detector(cache):
    current_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    if "SOLUTION":
        for layer in range(cfg["n_layers"]):
            attn = cache[f"blocks.{layer}.attn.hook_attn"]
            current_attn_score[layer] = reduce(
                attn.diagonal(dim1=-2, dim2=-1),
                "batch head_index pos -> head_index",
                "mean",
            )
        return current_attn_score


def prev_attn_detector(cache):
    prev_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    if "SOLUTION":
        for layer in range(cfg["n_layers"]):
            attn = cache[f"blocks.{layer}.attn.hook_attn"]
            prev_attn_score[layer] = reduce(
                attn.diagonal(dim1=-2, dim2=-1, offset=-1),
                "batch head_index pos -> head_index",
                "mean",
            )
        return prev_attn_score


def first_attn_detector(cache):
    first_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
    if "SOLUTION":
        for layer in range(cfg["n_layers"]):
            attn = cache[f"blocks.{layer}.attn.hook_attn"]
            first_attn_score[layer] = reduce(
                attn[:, :, :, 0], "batch head_index pos -> head_index", "mean"
            )
        return first_attn_score


def plot_head_scores(scores_tensor, title=""):
    px.imshow(
        to_numpy(scores_tensor),
        labels={"y": "Layer", "x": "Head"},
        title=title,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).show()


if MAIN:
    current_attn_scores = current_attn_detector(cache_2)
    plot_head_scores(current_attn_scores, "Current Token Heads")
    prev_attn_scores = prev_attn_detector(cache_2)
    plot_head_scores(prev_attn_scores, "Prev Token Heads")
    first_attn_scores = first_attn_detector(cache_2)
    plot_head_scores(first_attn_scores, "First Token Heads")

# %%
"""
## Ablations

An ablation is a simple causal intervention on a model - we pick some part of it and set it to zero. This is a crude proxy for how much that part matters. Further, if we have some story about how a specific circuit in the model enables some capability, showing that ablating *other* parts does nothing can be strong evidence of this.

This is normally a pain, but the hooking API makes it very easy! You'll need two new concepts:
* **Hook functions:** A hook function has the interface `def hook_fn(activation, hook)` and is associated with a particular activation. When the model is run, our hook function triggers, and is run on its activation. By default it just accesses the activation, but if the hook function edits the activation in place or returns a new one, the model then continues running with that new activation instead
    * You can ignore the `hook` argument for now
* We can temporarily add in a hook to a specific activation by defining this hook function, and then calling `logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_function)])`. This adds the hook for a single forward pass
    * Note that `fwd_hooks` needs to be a keyword argument

(If you want to be able to do this in Pyt generally, you'll need to learn to use [Pyt hooks](https://blog.paperspace.com/pyt-hooks-gradient-clipping-debugging/). These are really useful, but also a major headache and have a bad UI. Under the hood, my interface is just a nice wrapper around Pyt hooks that gets rid of a lot of the pain points and [I'm extending it to open source LLMs like GPT-2](https://colab.research.google.com/drive/1_tH4PfRSPYuKGnJbhC1NqFesOYuXrir_#scrollTo=zs8juArnyuyB) - feedback welcome!)
"""

if MAIN:
    print("As a reminder, here's the name and shape of each hook-able activation")
    print(
        f"The batch size is {tokens_2.size(0)} and the context length is {tokens_2.size(1)}"
    )
    for activation_name in cache_2:
        activation = cache_2[activation_name]
        print(f"Activation: {activation_name} Shape: {activation.shape}")

# %%
"""
Here's an example of using hooks to ablate the residual stream just before the logits after the first 3 tokens. We see that the per token loss is the same for the first 3 tokens, and wildly off for the rest. Unsurprisingly, ablating the entire residual stream harms performance!
"""


def ablate_residual_stream_hook(resid_post, hook):
    # resid_post.shape is [batch, position, d_model]
    resid_post[:, 3:] = 0.0
    return resid_post


def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    # Negate so high loss is bad, and index by zero to remove the trivial batch dimension
    return -pred_log_probs[0]


if MAIN:
    corrupted_logits = model.run_with_hooks(
        tokens_2, fwd_hooks=[("blocks.1.hook_resid_post", ablate_residual_stream_hook)]
    )
    clean_per_token_losses = per_token_losses(logits_2, tokens_2)
    corrupted_per_token_losses = per_token_losses(corrupted_logits, tokens_2)
    px.line(
        to_numpy((corrupted_per_token_losses - clean_per_token_losses)),
        title="Difference in per token loss",
    ).show()

# %%
"""
Now, try writing a function that can take in tokens, a layer and a head index, and return the logits of the model on those tokens with that head ablated.

Hint: You have multiple choices of which activation to ablate

We can now plot the increase of loss for each head, as another crude proxy for how much it matters. (Note - large increase is bad, and means the head was important)

Bonus Exercise: How would you expect this to compare to your direct logit attribution scores for heads in layer 0? For heads in layer 1? Plot a scatter plot and compare these results to your predictions
"""


def ablated_head_run(
    model: AttnOnlyTransformer, tokens: t.Tensor, layer: int, head_index: int
):
    if "SOLUTION":

        def ablate_head_hook(value, hook):
            value[:, :, head_index, :] = 0.0
            return value

        logits = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", ablate_head_hook)]
        )
        return logits


def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -(pred_log_probs.mean())


if MAIN:
    original_loss = cross_entropy_loss(logits_2, tokens_2)
    ablation_scores = t.zeros((cfg["n_layers"], cfg["n_heads"]))
    for layer in range(cfg["n_layers"]):
        for head_index in range(cfg["n_heads"]):
            ablation_scores[layer, head_index] = (
                cross_entropy_loss(
                    ablated_head_run(model, tokens_2, layer, head_index), tokens_2
                )
                - original_loss
            )
    plot_head_scores(ablation_scores)

# %%

"""
## Finding Induction Circuits
(Note: I use induction *head* to refer to the head in the second layer which attends to the 'token immediately after the copy of the current token', and induction *circuit* to refer to the circuit consisting of the composition of a previous token head in layer 0 and an induction head in layer 1)

[Induction heads](https://transformer-circuits.pub/2021/framework/index.html#induction-heads) are the first sophisticated circuit we see in transformers! And are sufficiently interesting that we wrote [another paper just about them](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

<details><summary>An aside on why induction heads are a big deal</summary>

There's a few particularly striking things about induction heads:

* They develop fairly suddenly in a phase change - from about 2B to 4B tokens we go from no induction heads to pretty well developed ones. This is a striking divergence from a 1L model [see the training curves for this model vs a 1L one](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-11-08-65---VmlldzoyNTI0MDQx?accessToken=extt248d3qoxbqw1zy05kplylztjmx2uqaui3ctqb0zruem0tkpwrssq2ao1su3j) and can be observed in much larger models (eg a 13B one)
    * Phase changes are particularly interesting (and depressing) from an alignment perspective, because the prospect of a sharp left turn, or emergent capabilities like deception or situational awareness seems like worlds where alignment may be harder, and we get caught by surprise without warning shots or simpler but analogous models to test our techniques on.

* They are responsible for a significant loss decrease - so much so that there's a visible bump in the loss curve when they develop (this change in loss can be pretty comparable to the increase in loss from major increases in model size, though this is hard to make an apples-to-apples comparison)
* They seem to be responsible for the vast majority of in-context learning - the ability to use far back tokens in the context to predict the next token. This is a significant way in which transformers outperform older architectures like RNNs or LSTMs, and induction heads seem to be a big part of this.
* The same core circuit seems to be used in a bunch of more sophisticated settings, such as translation or few-shot learning - there are heads that seem clearly responsible for those *and* which double as induction heads

</details>

We're going to spend the next two section applying the interpretability tools we just built to hunting down induction heads - first doing feature analysis to find the relevant heads and what they're doing, and then mechanistically reverse engineering the details of how the circuit works. I recommend you re-read the inductions head section of the paper or read [this intuitive explanation from Mary Phuong](https://docs.google.com/document/d/14HY2xKDW6Pup_-XNXQBjoYbc2xFILz06uxHkk9-pYmY/edit), but in brief, the induction circuit consists of a previous token head in layer 0 and an induction head in layer 1, where the induction head learns to attend to the token immediately *after* copies of the current token via K-Composition with the previous token head.

<p align="center">
    <img src="w2d4_Induction_Head_Pic.png" width="400" />
</p>

**Recommended Exercise:** Before continuing, take a few minutes to think about how you would implement an induction circuit if you were hand-coding the weights of an attention-only transformer:
* How would you implement a copying head?
* How would you implement a previous token head?
* How would you implement an induction head?


<details><summary>My summary of the algorithm</summary>

* Head L0H7 is a previous token head (the QK-circuit ensures it always attends to the previous token).
* The OV circuit of head L0H7 writes a copy of the previous token in a *different* subspace to the one used by the embedding.
* The output of head L0H7 is used by the *key* input of head L1H4 via K-Composition to attend to 'the source token whose previous token is the destination token'.
* The OV-circuit of head L1H4 copies the *value* of the source token to the same output logit
    * Note that this is copying from the embedding subspace, *not* the L0H7 output subspace - it is not using V-Composition at all

To emphasise - the sophisticated hard part is computing the *attention* pattern of the induction head - this takes careful composition. The previous token and copying parts are fairly easy. This is a good illustrative example of how the QK circuits and OV circuits act semi-independently, and are often best thought of somewhat separately. And that computing the attention patterns can involve real and sophisticated computation!
</details>

<details>

<summary>Exercise: Why couldn't an induction head form in a 1L model?</summary>

Because this would require a head which attends a key position based on the *value* of the token before it. Attention scores are just a function of the key token and the query token, and are not a function of other tokens.
The attention pattern *does* allow other tokens because of softmax - if another key token has a high attention score, softmax inhibits this pair. But this inhibition is symmetric across positions, so can't systematically favour the token *next* to the relevant one.

Note that a key detail is that the value of adjacent tokens are (approximately) unrelated - if the model wanted to attend based on relative *position* this is easy.
</details>
"""

# %%
"""
## Checking for the induction capability

A striking thing about models with induction heads is that, given a repeated sequence of random tokens, they can predict the repeated half of the sequence. This is nothing like it's training data, so this is kind of wild! The ability to predict this kind of out of distribution generalisation is a strong point of evidence that you've really understood a circuit.

To check that this model has induction heads, we're going to run it on exactly that, and compare performance on the two halves - you should see a striking difference in the per token losses

<details><summary>Tips:</summary>

Remember to include a cache called `rep_cache` and to reset the model hooks afterwards, and to include the prefix array at the start. Your `repeated_tokens` array should have shape `[batch, 1+2*seq_len]`</details>
"""


def run_and_cache_model_repeated_tokens(
    model, seq_len, batch=1
) -> tuple[t.Tensor, t.Tensor, dict]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Add a prefix token, since the model was always trained to have one.

    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: {} The cache of the model run on rep_tokens
    """
    prefix = t.ones((batch, 1), dtype=t.int64) * tokenizer.bos_token_id
    "SOLUTION"
    rep_cache = {}
    model.cache_all(rep_cache)
    rand_tokens = t.randint(1000, 10000, (batch, seq_len))
    rep_tokens = t.concat([prefix, rand_tokens, rand_tokens], dim=1).cuda()
    rep_logits = model(rep_tokens)
    return rep_logits, rep_tokens, rep_cache


if MAIN:
    """
    These are small numbers, since the results are very obvious and this makes it easier to visualise - in practice we'd obviously use larger ones on more subtle tasks. But it's often easiest to iterate and debug on small tasks.
    """
    seq_len = 50
    batch = 1
    rep_logits, rep_tokens, rep_cache = run_and_cache_model_repeated_tokens(
        model, seq_len, batch
    )
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print("Performance on the first half:", ptl[:seq_len].mean())
    print("Performance on the second half:", ptl[seq_len:].mean())
    px.line(
        to_numpy(ptl),
        hover_name=to_numpy(rep_tokens[0, :-1]),
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
    ).show()

# %%
"""
## Looking for Induction Attention Patterns

The next natural thing to check for is the induction attention pattern.

First, visualise the attention patterns for each head and manually check for likely heads in the second layer. Hint: Reuse the `plot_attn_pattern` function, and remember to remove the batch dimension!

<details><summary>What you should see</summary>

You should see that heads 4 and 10 are strongly induction-y, and the rest aren't
</details>
"""
if MAIN:
    if "SOLUTION":
        plot_attn_pattern(
            rep_cache["blocks.1.attn.hook_attn"][0],
            rep_tokens,
            "Layer 1 attention on repeated sequence",
        )

# %%
"""
Next, make an induction pattern score function, which looks for the average attention paid to the offset diagonal. Do this in the same style as our earlier head scorers.

<details><summary>Gotcha</summary> The offset in pattern.diagonal should be -(seq_len-1)</details>
"""


def induction_attn_detector(cache):
    if "SOLUTION":
        induction_attn_score = t.zeros(cfg["n_layers"], cfg["n_heads"])
        for layer in range(cfg["n_layers"]):
            attn = cache[f"blocks.{layer}.attn.hook_attn"]
            induction_attn_score[layer] = reduce(
                attn.diagonal(dim1=-2, dim2=-1, offset=-(seq_len - 1)),
                "batch head_index pos -> head_index",
                "mean",
            )
        return induction_attn_score


if MAIN:
    induction_attn_scores = induction_attn_detector(rep_cache)
    plot_head_scores(induction_attn_scores)

# %%
"""
### Logit Attribution

We can reuse our `logit_attribution` function from earlier to look at the contribution to the correct logit from each term on the first and second half of the sequence.

Gotchas:
* Remember to remove the batch dimension
* Remember to split the sequence in two, with one overlapping token (since predicting the next token involves removing the final token with no label) - your logit_attrs should both have shape [seq_len, 2*n_heads + 1] (ie [50, 25] here)

Note that the first plot will be pretty meaningless (why?)
"""

if MAIN:
    embed = rep_cache["hook_embed"]
    l1_results = rep_cache["blocks.0.attn.hook_result"]
    l2_results = rep_cache["blocks.1.attn.hook_result"]
    first_half_tokens = rep_tokens[0, : 1 + seq_len]
    second_half_tokens = rep_tokens[0, seq_len:]
    if "SOLUTION":
        first_half_logit_attr = logit_attribution(
            embed[0, : 1 + seq_len],
            l1_results[0, : 1 + seq_len],
            l2_results[0, : 1 + seq_len],
            model.unembed.W_U,
            first_half_tokens,
        )
        second_half_logit_attr = logit_attribution(
            embed[0, seq_len:],
            l1_results[0, seq_len:],
            l2_results[0, seq_len:],
            model.unembed.W_U,
            second_half_tokens,
        )
    plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    plot_logit_attribution(second_half_logit_attr, second_half_tokens)


# %%
"""
## Ablations

We can re-use our `ablated_head_run` function from earlier to ablate each head and compare the change in loss. (Note again that positive change in loss is bad)

Exercise: Before running this, what do you predict will happen? In particular, which cells will be significant?

<details><summary>Bonus exercise</summary>

Try ablating *every* head apart from the previous token head and the two induction heads. What does this do to performance? What if you mean ablate it, rather than zero ablating it?
</details>
"""

if MAIN:
    original_rep_loss = cross_entropy_loss(rep_logits, rep_tokens)
    ablation_scores = t.zeros((cfg["n_layers"], cfg["n_heads"]))
    if "SOLUTION":
        for layer in range(cfg["n_layers"]):
            for head_index in range(cfg["n_heads"]):
                ablation_scores[layer, head_index] = (
                    cross_entropy_loss(
                        ablated_head_run(model, rep_tokens, layer, head_index),
                        rep_tokens,
                    )
                    - original_rep_loss
                )
    plot_head_scores(ablation_scores)


# %%
"""
## Previous Token Head

To identify the previous token head, we can reuse our `prev_attn_detector` function, and then visualise the layer 0 attention patterns to validate
"""

if MAIN:
    if "SOLUTION":
        prev_attn_scores = prev_attn_detector(rep_cache)
        plot_head_scores(prev_attn_scores)
        plot_attn_pattern(rep_cache["blocks.0.attn.hook_attn"][0], rep_tokens)

# %%
"""
## Mechanistic Analysis of Induction Circuits

Most of what we did above was feature analysis - we looked at activations (here just attention patterns) and tried to interpret what they were doing. Now we're going to do some mechanistic analysis - digging into the weights and using them to reverse engineer the induction head algorithm and verify that it is really doing what we think it is.

### Reverse Engineering OV-Circuit Analysis

Let's start with an easy parts of the circuit - the copying OV circuit of L1H4 and L1H10. Let's start with head 4. The only interpretable things here are the input tokens and output logits, so we want to study the factored matrix W_UW_OW_VW_E. Let's begin by calculating it.

You should get a matrix OV_circuit with shape [d_vocab, d_vocab]

Exercise: What does this matrix represent, conceptually?

Tip: If you start running out of CUDA memory, cast everything to float16 (tensor -> tensor.half()) before multiplying - 50K x 50K matrices are large! Alternately, copy it to CPU and delete it from CUDA
"""
if MAIN:
    head_index = 4
    layer = 1
    if "SOLUTION":
        W_O = model.blocks[layer].attn.W_O[head_index]
        W_V = model.blocks[layer].attn.W_V[head_index]
        W_E = model.embed.W_E
        W_U = model.unembed.W_U
        OV_circuit = W_U.half() @ W_O.half() @ W_V.half() @ W_E.half()

# %%
"""
Now we want to check that this matrix is the identity. This is a surprisingly big pain! It's a 50K x 50K matrix, which is far too big to visualise. And in practice, this is going to be fairly noisy. And we don't strictly need to get it to be the identity, just have big terms along the diagonal.

First, to validate that it looks diagonal-ish, let's pick 200 random rows and columns and visualise that - it should at least look identity-ish here!
"""

if MAIN:
    rand_indices = t.randperm(cfg["d_vocab"])[:200]
    px.imshow(to_numpy(OV_circuit[rand_indices][:, rand_indices])).show()

# %%
"""
Now we want to try to make a summary statistic to capture this. Accuracy is a good one - what fraction of the time is the largest logit in a column on the diagonal?

Bonus exercise: Top-5 accuracy is also a good metric (use `t.topk`, take the indices output)

When I run this I get about 30.8% - pretty underwhelming. It goes up to 47.72% for top-5. What's up with that?
"""


def top_1_acc(OV_circuit):
    """
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    """
    "SOLUTION"
    return (OV_circuit.argmax(0) == t.arange(cfg["d_vocab"]).cuda()).sum() / cfg[
        "d_vocab"
    ]


if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc(OV_circuit))

# %%
"""
Now we return to why we have *two* induction heads. If both have the same attention pattern, the effective OV circuit is actually W_U(W_O[4]W_V[4]+W_O[10]W_V[10])W_E, and this is what matters. So let's calculate this and re-run our analysis on that!
<details>

<summary>Exercise: Why might the model want to split the circuit across two heads?</summary>

Because W_OW_V is a rank 64 matrix. The sum of two is a rank 128 matrix. This can be a significantly better approximation to the desired 50K x 50K matrix!
</details>

"""

if MAIN:
    if "SOLUTION":
        W_OV_full = (
            model.blocks[1].attn.W_O[4] @ model.blocks[1].attn.W_V[4]
            + model.blocks[1].attn.W_O[10] @ model.blocks[1].attn.W_V[10]
        )
        OV_circuit_full = W_U.half() @ W_OV_full.half() @ W_E.half()
    print("Top 1 accuracy for the full OV Circuit:", top_1_acc(OV_circuit_full))

    try:
        del OV_circuit_full
        del OV_circuit
    except:
        pass


# %%
"""
### Reverse Engineering Positional Embeddings + Prev Token Head

The other easy circuit is the QK-circuit of L0H7 - how does it know to be a previous token circuit?

We can multiply out the full QK circuit via the positional embeddings: W_pos.T W_Q.T W_K W_pos to get a matrix pos_by_pos of shape [max_ctx, max_ctx]

We can then mask it and apply a softmax, and should get a clear stripe on the lower diagonal (Tip: Click and drag to zoom in, hover over cells to see their values and indices!)

Hints:
* Remember to divide by sqrt(d_head)!
* Reuse the `mask_scores` from earlier

(Note: If we were being properly rigorous, we'd also need to show that the token embedding wasn't important for the attention scores.)
"""

if MAIN:
    if "SOLUTION":
        layer = 0
        head_index = 7
        W_Q = model.blocks[layer].attn.W_Q[head_index]
        W_K = model.blocks[layer].attn.W_K[head_index]
        W_pos = model.pos_embed.W_pos
        pos_by_pos = (W_pos.T @ W_Q.T @ W_K @ W_pos) / np.sqrt(cfg["d_head"])
        pos_by_pos_pattern = F.softmax(mask_scores(pos_by_pos), dim=-1)
    px.imshow(
        to_numpy(pos_by_pos_pattern[:200, :200]),
        labels={"y": "Query", "k": "Key"},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0.0,
    ).show()


# %%
r"""

### Composition Analysis

We now dig into the hard part of the circuit - demonstrating the K-Composition between the previous token head and the induction head.

#### Splitting activations

We can repeat the trick from the logit attribution scores. The qk_input for layer 1 is the sum of 14 terms (2+n_heads) - the embedding, the positional embedding, and the results of each layer 0 head. So for each head in layer 1, the query tensor (ditto key) is:
`W_Q @ qk_input == W_Q @ (embed + pos_embed + \sum result_i) == W_Q @ embed + W_Q @ pos_embed + \sum W_Q @ result_i`

We can now analyse the relative importance of these terms! A very crude measure is to take the norm of each term (by component and position) - when we do this here, we show clear dominance in the k from L0H7, and in the q from the embed (and pos embed).

Note that this is a pretty dodgy metric - q and k are not inherently interpretable! But it can be a good and easy to compute proxy.
"""


def decompose_qk_input(cache: dict) -> t.Tensor:
    """
    Output is decomposed_qk_input, with shape [2+num_heads, position, d_model]
    """
    "SOLUTION"
    embed: t.Tensor = cache["hook_embed"][batch_index]
    pos_embed: t.Tensor = cache["hook_pos_embed"]
    head_results: t.Tensor = cache["blocks.0.attn.hook_result"]
    head_results = rearrange(
        head_results, "1 position head_index d_model -> head_index position d_model"
    )
    return t.concat([embed[None, :], pos_embed[None, :], head_results], dim=0)  # type: ignore


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    """
    Output is decomposed_q with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just q)
    """
    "SOLUTION"
    W_Q = model.blocks[1].attn.W_Q[ind_head_index]
    decomposed_q = decomposed_qk_input @ W_Q.T
    return decomposed_q


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    """
    Output is decomposed_k with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just k) - exactly analogous as for q
    """
    "SOLUTION"
    W_K = model.blocks[1].attn.W_K[ind_head_index]
    decomposed_k = decomposed_qk_input @ W_K.T
    return decomposed_k


if MAIN:
    batch_index = 0
    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    assert t.isclose(
        decomposed_qk_input.sum(0),
        rep_cache["blocks.1.attn.hook_qk_input"][0],
        rtol=1e-2,
        atol=1e-5,
    ).all()
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    assert t.isclose(
        decomposed_q.sum(0),
        rep_cache["blocks.1.attn.hook_q"][0, :, ind_head_index],
        rtol=1e-2,
        atol=1e-3,
    ).all()
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    assert t.isclose(
        decomposed_k.sum(0),
        rep_cache["blocks.1.attn.hook_k"][0, :, ind_head_index],
        rtol=1e-2,
        atol=1e-2,
    ).all()

    component_labels = ["Embed", "PosEmbed"] + [
        f"L0H{h}" for h in range(cfg["n_heads"])
    ]
    px.imshow(
        to_numpy(decomposed_q.pow(2).sum([-1])),
        color_continuous_scale="Blues",
        labels={"x": "Pos", "y": "Component"},
        y=component_labels,
        title="Norms of components of query",
    ).show()
    px.imshow(
        to_numpy(decomposed_k.pow(2).sum([-1])),
        color_continuous_scale="Blues",
        labels={"x": "Pos", "y": "Component"},
        y=component_labels,
        title="Norms of components of key",
    ).show()

# %%
"""
We can do one better, and take the decomposed attention scores. This is a bilinear function of q and k, and so we will end up with a decomposed_scores tensor with shape [query_component, key_component, query_pos, key_pos], where summing along BOTH of the first axes will give us the original attention scores (pre-mask)

Implement the function giving the decomposed scores (remember to scale by sqrt(d_head)!) For now, don't mask it.

We can now look at the standard deviation across the key and query positions for each pair of components. This is a proxy for 'how much the attention pattern depends on that component for the query and for the key. And we can plot a num_components x num_components heatmap to see how important each pair is - this again clearly shows the pair of Q=Embed, K=L0H7 dominates.

We can even plot the attention scores for that component and see a clear induction stripe.

<details>
<summary>Exercise: Do you expect this to be symmetric? Why/why not?</summary>

No, because the y axis is the component in the *query*, the x axis is the component in the *key* - these are not symmetric!
</details>

<details>
<summary>Exercise: Why do I focus on the attention scores, not the attention pattern? (Ie pre softmax not post softmax)</summary>

Because the decomposition trick *only* works for things that are linear - softmax isn't linear and so we can no longer consider each component independently
</details>
"""


def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    "SOLUTION"
    decomposed_scores = t.einsum("cqh,Ckh->cCqk", decomposed_q, decomposed_k) / np.sqrt(
        cfg["d_head"]
    )
    return decomposed_scores


if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = reduce(
        decomposed_scores,
        "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp",
        t.std,
    )
    px.imshow(
        to_numpy(decomposed_stds),
        labels={"x": "Key Component", "y": "Query Component"},
        x=component_labels,
        y=component_labels,
        color_continuous_scale="Blues",
        title="Standard deviations of components of scores",
    ).show()
    plot_attn_pattern(
        t.tril(decomposed_scores[0, 9]),
        rep_tokens,
        title="Attention Scores for component from Q=Embed and K=Prev Token Head",
    )

# %%
"""
#### Interpreting the K-Composition Circuit
Now we know that head L1H4 is composing with head L0H7 via K composition, we can multiply through to create a full end-to-end circuit W_E.T @ W_Q[1, 4].T @ W_K[1, 4] @ W_O[0, 7] @ W_V[0, 7] @ W_E and verify that it's the identity.

We can now reuse our `top_1_acc` code from before to check that it's identity-like, we see that half the time the diagonal is the top (goes up to 89% with top 5 accuracy) (We transpose first, because we want the argmax over the key dimension)

Remember to cast to float16 (tensor -> tensor.half()) to stop your GPU getting too full!
"""


def find_K_comp_full_circuit(prev_token_head_index, ind_head_index):
    """
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    """
    "SOLUTION"
    W_E = model.embed.W_E
    W_OV = (
        model.blocks[0].attn.W_O[prev_token_head_index]
        @ model.blocks[0].attn.W_V[prev_token_head_index]
    )
    W_QK = (
        model.blocks[1].attn.W_Q[ind_head_index].T
        @ model.blocks[1].attn.W_K[ind_head_index]
    )
    return t.einsum("mv,mM,Mn,nV->vV", W_E.half(), W_QK.half(), W_OV.half(), W_E.half())


if MAIN:
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_circuit(prev_token_head_index, ind_head_index)
    print(
        "Fraction of tokens where the highest activating key is the same token",
        top_1_acc(K_comp_circuit.T).item(),
    )
    del K_comp_circuit

# %%
r"""
## Further Exploration of Induction Circuits

I now consider us to have fully reverse engineered an induction circuit - by both interpreting the features and by reverse engineering the circuit from the weights. But there's a bunch more ideas that we can apply for finding circuits in networks that are fun to practice on induction heads, so here's some bonus content - feel free to skip to the later bonus ideas though.

### Composition scores

A particularly cool idea in the paper is the idea of [virtual weights](https://transformer-circuits.pub/2021/framework/index.html#residual-comms), or compositional scores. (though I came up with it, so I'm deeply biased) This is used [to identify induction heads](https://transformer-circuits.pub/2021/framework/index.html#analyzing-a-two-layer-model)

The key idea of compositional scores is that the residual stream is a large space, and each head is reading and writing from small subspaces. By defaults, any two heads will have little overlap between their subspaces (in the same way that any two random vectors have almost zero dot product in a large vector space). But if two heads are deliberately composing, then they will likely want to ensure they write and read from similar subspaces, so that minimal information is lost. As a result, we can just directly look at "how much overlap there is" between the output space of the earlier head and the K, Q, or V input space of the later head. We represent the output space with $W_OV=W_OW_V$, and the input space with $W_QK^T=W_K^TW_Q$ (for Q-composition), $W_QK=W_Q^TW_K$ (for K-Composition) or $W_OV=W_OW_V$ (for V-Composition, of the later head). Call these matrices $W_A$ and $W_B$ respectively.

How do we formalise overlap? This is basically an open question, but a surprisingly good metric is $\frac{|W_BW_A|}{|W_B||W_A|}$ where $|W|=\sum_{i,j}W_{i,j}^2$ is the Frobenius norm, the sum of squared elements. Let's calculate this metric for all pairs of heads in layer 0 and layer 1 for each of K, Q and V composition and plot it.

<details><summary>Why do we use W_OV as the output weights, not W_O? (and W_QK not W_Q or W_K, etc)</summary>

Because W_O is arbitrary - we can apply an arbitrary invertible matrix to W_O and its inverse to W_V and preserve the product W_OV. Though in practice, it's an acceptable approximation.
</details>

"""


def frobenius_norm(tensor):
    """
    Implicitly allows batch dimensions
    """
    return tensor.pow(2).sum([-2, -1])


def get_q_comp_scores(W_QK, W_OV):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the Q-Composition score from head L0Hj to L1Hi
    """
    "SOLUTION"
    q_full = t.einsum("Imn,imM->IinM", W_QK, W_OV)
    comp_scores = (
        frobenius_norm(q_full)
        / frobenius_norm(W_QK)[:, None]
        / frobenius_norm(W_OV)[None, :]
    )
    return comp_scores


def get_k_comp_scores(W_QK, W_OV):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the K-Composition score from head L0Hj to L1Hi
    """
    "SOLUTION"
    k_full = t.einsum("Inm,imM->IinM", W_QK, W_OV)
    comp_scores = (
        frobenius_norm(k_full)
        / frobenius_norm(W_QK)[:, None]
        / frobenius_norm(W_OV)[None, :]
    )
    return comp_scores


def get_v_comp_scores(W_OV_1, W_OV_0):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the V-Composition score from head L0Hj to L1Hi
    """
    "SOLUTION"
    v_full = t.einsum("Inm,imM->IinM", W_OV_1, W_OV_0)
    comp_scores = (
        frobenius_norm(v_full)
        / frobenius_norm(W_OV_1)[:, None]
        / frobenius_norm(W_OV_0)[None, :]
    )
    return comp_scores


if MAIN:
    W_O = model.blocks[0].attn.W_O
    W_V = model.blocks[0].attn.W_V
    W_OV_0 = t.einsum("imh,ihM->imM", W_O, W_V)
    W_Q = model.blocks[1].attn.W_Q
    W_K = model.blocks[1].attn.W_K
    W_V = model.blocks[1].attn.W_V
    W_O = model.blocks[1].attn.W_O
    W_QK = t.einsum("ihm,ihM->imM", W_Q, W_K)
    W_OV_1 = t.einsum("imh,ihM->imM", W_O, W_V)

    q_comp_scores = get_q_comp_scores(W_QK, W_OV_0)
    k_comp_scores = get_k_comp_scores(W_QK, W_OV_0)
    v_comp_scores = get_v_comp_scores(W_OV_1, W_OV_0)
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()

# %%
"""
#### Setting a Baseline

To interpret the above graphs we need a baseline! A good one is what the scores look like at initialisation. Make a function that randomly generates a composition score 200 times and tries this. Remember to generate 4 [d_head, d_model] matrices, not 2 [d_model, d_model] matrices! This model was initialised with Kaiming Uniform Initialisation:

```python
W = t.empty(shape)
nn.init.kaiming_uniform_(W, a=np.sqrt(5))
```

(Ideally we'd do a more efficient generation involving batching, and more samples, but we won't worry about that here)
"""


def generate_single_random_comp_score() -> float:
    """
    Write a function which generates a single composition score for random matrices
    """
    "SOLUTION"
    matrices = [t.empty((cfg["d_head"], cfg["d_model"])) for i in range(4)]
    for mat in matrices:
        nn.init.kaiming_uniform_(mat, a=np.sqrt(5))
    W1 = matrices[0].T @ matrices[1]
    W2 = matrices[2].T @ matrices[3]
    W3 = W1 @ W2
    return (frobenius_norm(W3) / frobenius_norm(W1) / frobenius_norm(W2)).item()


if MAIN:
    comp_scores_baseline = np.array(
        [generate_single_random_comp_score() for i in range(200)]
    )
    print("Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()
# %%
"""
We can re-plot our above graphs with this baseline set to white. Look for interesting things in this graph!
"""
if MAIN:
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()

# %%
"""
#### Theory + Efficient Implementation

So, what's up with that metric? The key is a cute linear algebra result that the Frobenius norm is equal to the sum of the squared singular values.

<details>
<summary>Proof</summary>
M = USV in the singular value decomposition. U and V are rotations and do not change norm, so |M|=|S|
</details>

So if $W_A=U_AS_AV_A$, $W_B=U_BS_BV_B$, then $|W_A|=|S_A|$, $|W_B|=|S_B|$ and $|W_AW_B|=|S_AV_AU_BS_B|$. In some sense, $V_AU_B$ represents how aligned the subspaces written to and read from are, and the $S_A$ and $S_B$ terms weights by the importance of those subspaces.

We can also use this insight to write a more efficient way to calculate composition scores - this is extremely useful if you want to do this analysis at scale! The key is that we know that our matrices have a low rank factorisation, and it's much cheaper to calculate the SVD of a narrow matrix than one that's large in both dimensions. See the [algorithm described at the end of the paper](https://transformer-circuits.pub/2021/framework/index.html#induction-heads:~:text=Working%20with%20Low%2DRank%20Matrices) (search for SVD). Go implement it!


Gotcha: Note that `torch.svd(A)` returns `(U, S, V.T)` not `(U, S, V)`

Bonus exercise: Write a batched version of this that works for batches of heads, and run this over GPT-2 - this should be doable for XL, I think.

"""


def stranded_svd(A: t.Tensor, B: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Returns the SVD of AB in the torch format (ie (U, S, V^T))
    """
    "SOLUTION"
    UA, SA, VhA = t.svd(A)
    UB, SB, VhB = t.svd(B)
    intermed = SA.diag() @ VhA.T @ UB @ SB.diag()
    UC, SC, VhC = t.svd(intermed)
    return (UA @ UC), SC, (VhB @ VhC).T


def stranded_composition_score(
    W_A1: t.Tensor, W_A2: t.Tensor, W_B1: t.Tensor, W_B2: t.Tensor
):
    """
    Returns the composition score for W_A = W_A1 @ W_A2 and W_B = W_B1 @ W_B2, with the entries in a low-rank factored form
    """
    "SOLUTION"
    UA, SA, VhA = stranded_svd(W_A1, W_A2)
    UB, SB, VhB = stranded_svd(W_B1, W_B2)
    normA = SA.pow(2).sum()
    normB = SB.pow(2).sum()
    intermed = SA.diag() @ VhA.T @ UB @ SB.diag()
    UC, SC, VhC = t.svd(intermed)
    return SC.pow(2).sum() / normA / normB


# %%
"""
#### Targeted Ablations

We can refine the ablation technique to detect composition by looking at the effect of the ablation on the attention pattern of an induction head, rather than the loss. Let's implement this!

Gotcha - by default, run_with_hooks removes any existing hooks when it runs, if you want to use caching set the reset_hooks_start flag to False
"""


def ablation_induction_score(prev_head_index: int, ind_head_index: int) -> t.Tensor:
    """
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    """

    def ablation_hook(v, hook):
        v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            ("blocks.0.attn.hook_v", ablation_hook),
            ("blocks.1.attn.hook_attn", induction_pattern_hook),
        ],
    )
    return model.blocks[1].attn.hook_attn.ctx[prev_head_index]


if MAIN:
    for i in range(cfg["n_heads"]):
        print(f"Ablation effect of head {i}:", ablation_induction_score(i, 4).item())


# %%
"""
# Bonus

## Looking for Circuits in Real LLMs

A particularly cool application of these techniques is looking for real examples of circuits in large language models. Fortunately, there's a bunch of open source ones we can play around with! I've made a library for transformer interpretability called EasyTransformer. It loads in an open source LLMs into a simplified transformer, and gives each activation a unique name. With this name, we can set a hook that accesses or edits that activation, with the same API that we've been using on our 2L Transformer. You can see it in `w2d4_easy_transformer.py` - feedback welcome!

**Example:** Ablating the 5th attention head in layer 4 of GPT-2 medium
"""

from w2d4_easy_transformer import EasyTransformer

model = EasyTransformer("gpt2-medium")
text = "Hello world"
input_tokens = model.to_tokens(text)

head_index = 5
layer = 4


def ablation_hook(value, hook):
    # value is the value activation for this attention layer
    # It has shape [batch, position, head_index, d_head]
    value[:, :, head_index, :] = 0.0
    return value


logits = model.run_with_hooks(
    input_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", ablation_hook)]
)

"""
This library should make it moderately easy to play around with these models - I recommend going wild and looking for interesting circuits!

This part of the day is deliberately left as an unstructured bonus, so I recommend following your curiosity! But if you want a starting point, here are some suggestions:
- Look for induction heads - try repeating all of the steps from above. Do they follow the same algorithm?
- Look for neurons that erase info
    - Ie having a high negative cosine similarity between the input and output weights
- Try to interpret a position embedding
<details><summary>Positional Embedding Hint:</summary>

Look at the singular value decomposition `t.svd` and plot the principal components over position space. High ones tend to be sine and cosine waves of different frequencies.

**Gotcha:** The output of `t.svd` is `U, S, Vh = t.svd(W_pos)`, where `U @ S.diag() @ Vh.T == W_pos` - W_pos has shape [d_model, n_ctx], so the ith principal component on the n_ctx side is `W_pos[:, i]` NOT `W_pos[i, :]
</details>

- Look for heads with interpretable attention patterns: Eg heads that attend to the same word (or subsequent word) when given text in different languages, or the most recent proper noun, or the most recent full-stop, or the subject of the sentence, etc.
    - Pick a head, ablate it, and run the model on a load of text with and without the head. Look for tokens with the largest difference in loss, and try to interpret what the head is doing.
- Try replicating some of Kevin's work on indirect object vs
- Inspired by the [ROME paper](https://rome.baulab.info/), use the causal tracing technique of patching in residual stream - can you analyse how the network answers different facts?

Note: I apply several simplifications to the resulting transformer - these leave the model mathematically equivalent and doesn't change the output log probs, but does somewhat change the structure of the model and one change translates the output logits by a constant - see [Discussion](https://colab.research.google.com/drive/1_tH4PfRSPYuKGnJbhC1NqFesOYuXrir_#scrollTo=Discussion) for some discussion of these.

## Training Your Own Toy Models

A fun exercise is training models on the minimal task that'll produce induction heads - predicting the next token in a sequence of random tokens with repeated subsequences. You can get a small 2L Attention-Only model to do this.

<details>
<summary>Tips</summary>

* Make sure to randomise the positions that are repeated! Otherwise the model can just learn the boring algorithm of attending to fixed positions
* It works better if you *only* evaluate loss on the repeated tokens, this makes the task less noisy.
* It works best with several repeats of the same sequence rather than just one.
* If you do things right, and give it finite data + weight decay, you *should* be able to get it to grok - this may take some hyper-parameter tuning though.
* When I've done this I get weird franken-induction heads, where each head has 1/3 of an induction stripe, and together cover all tokens.
* It'll work better if you only let the queries and keys access the positional embeddings, but *should* work either way
</details>

## Interpreting Induction Heads During Training

A particularly striking result about induction heads is that they consistently [form very abruptly in training as a phase change](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html#argument-phase-change), and are such an important capability that there is a [visible non-convex bump in the loss curve](https://wandb.ai/mechanistic-interpretability/attn-only/reports/loss_ewma-22-08-24-22-08-00---VmlldzoyNTI2MDM0?accessToken=r6v951q0e1l4q4o70wb2q67wopdyo3v69kz54siuw7lwb4jz6u732vo56h6dr7c2) (in this model, approx 2B to 4B tokens). I have a bunch of checkpoints for this model, you can try re-running the induction head detection techniques on intermediate checkpoints and see what happens. (Bonus points if you have good ideas for how to efficiently send you a bunch of 300MB checkpoints from Wandb lol)
"""

# %%
