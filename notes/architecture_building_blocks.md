# Model architecture building blocks – mentoring notes

Short notes we expand on in sessions. Each block = *what problem it solves* + *the core equation*.

## 1. Linear layer + non-linearity
`y = σ(Wx + b)`. Without the non-linearity σ (ReLU, GELU) any stack of layers collapses to one linear map.

## 2. Convolution
Weight sharing + locality: the same small kernel slides over the image, so the number of parameters
does not depend on image size and the layer is translation-equivariant. Stacking convs grows the
**receptive field**; pooling/stride trades spatial resolution for more channels (more "concepts").
See `src/conv_from_scratch.py`.

## 3. Normalisation (BatchNorm / LayerNorm)
Re-centres and re-scales activations: `x̂ = (x − μ)/σ · γ + β`. Stabilises training and allows
higher learning rates. BatchNorm normalises over the batch (CNNs); LayerNorm over features
(Transformers, works with batch size 1).

## 4. Residual block (ResNet, 2015)
`y = x + F(x)`. The layer only has to learn a *correction* F. Gradients flow straight through the
identity path (`∂y/∂x = I + ∂F/∂x`), which is why 100+ layer networks became trainable.
Typical CNN form: `conv → BN → ReLU → conv → BN`, add skip, `ReLU`. If shapes change, the skip
uses a 1×1 conv.

## 5. Attention (Transformer, 2017)
`Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V`
Each token builds a query, compares it to every key, and takes a weighted average of values →
global, content-dependent mixing (vs. the fixed local mixing of a conv).
* **Multi-head**: several attention maps in parallel, concatenated.
* **Transformer block**: `x = x + MHA(LN(x))`, then `x = x + MLP(LN(x))` – residuals + norm again.
* Cost is O(n²) in sequence length → motivation for windowed/linear attention.

## 6. Vision Transformer (ViT)
Cut the image into 16×16 patches → linear projection to tokens → add position embeddings →
standard Transformer encoder → classify from a [CLS] token. Weak inductive bias, so it needs more
data or pre-training than a CNN; hybrids (ConvNeXt, Swin) borrow ideas both ways.

## 7. Encoder–decoder / U-Net
Downsample to get context, upsample to recover resolution, with skip connections between
matching levels. Backbone of segmentation and diffusion models.

## Design checklist we use in sessions
1. What is the inductive bias the data needs (locality, order, permutation invariance)?
2. Where does information flow — is there a short gradient path (residuals)?
3. What normalisation and initialisation keep activations in a healthy range?
4. Parameter count / FLOPs vs. available data and hardware.
5. Ablate one change at a time and measure.
