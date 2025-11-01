# 📚 Generative Models — Learning Timeline: From-Scratch PyTorch Implementations

---

## 1. **Variational Autoencoder (VAE, 2013)**

* **Key innovation:** Reparameterization trick, ELBO training.
* **Learning focus:** Probabilistic encoders/decoders, KL divergence, latent traversals.
* **Paper:** [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114)
* **Datasets:** MNIST → FashionMNIST → CIFAR-10
* **Difficulty:** easy → moderate
* **Project tasks:** implement Gaussian encoder/decoder, ELBO loss, latent interpolation visualization.

---

## 2. **Autoregressive Models**

### 2a. **PixelRNN (2016)**

* **Key innovation:** Pixel-by-pixel autoregressive modeling.
* **Learning focus:** RNN-based factorization of pixels.
* **Paper:** [PixelRNN](https://proceedings.mlr.press/v48/oord16.pdf)
* **Datasets:** MNIST, CIFAR-10
* **Difficulty:** moderate
* **Project tasks:** implement sequential pixel prediction with RNNs.

### 2b. **PixelCNN (2016)**

* **Key innovation:** Convolutional autoregressive modeling; masked convolutions.
* **Learning focus:** Parallelized pixel prediction.
* **Paper:** [PixelCNN](https://proceedings.mlr.press/v48/oord16.pdf)
* **Datasets:** MNIST, CIFAR-10
* **Difficulty:** moderate
* **Project tasks:** implement masked convolutions, train on pixel likelihood, sampling loop.

---

## 3. **Flow-Based Models**

### 3a. **RealNVP (2016)**

* **Key innovation:** Invertible transformations, tractable log-likelihood.
* **Learning focus:** Affine coupling, log-determinants, forward/inverse mapping.
* **Paper:** [RealNVP](https://arxiv.org/abs/1605.08803)
* **Datasets:** CIFAR-10, low-res ImageNet
* **Difficulty:** moderate → hard
* **Project tasks:** implement coupling layers, compute Jacobian log-determinants.

### 3b. **Glow (2018)**

* **Key innovation:** Invertible 1×1 convolution, actnorm.
* **Learning focus:** Scaling flows to high-resolution images.
* **Paper:** [Glow](https://arxiv.org/pdf/1807.03039)
* **Datasets:** CIFAR-10, ImageNet
* **Difficulty:** hard
* **Project tasks:** implement invertible convolutions, multi-scale architecture.

---

## 4. **Generative Adversarial Networks (GANs)**

### 4a. **Vanilla GAN (2014)**

* **Key innovation:** Adversarial generator/discriminator.
* **Learning focus:** Minimax optimization, basic GAN training.
* **Paper:** [GAN](https://arxiv.org/pdf/1406.2661)
* **Datasets:** MNIST, CIFAR-10
* **Difficulty:** moderate
* **Project tasks:** implement basic GAN, train generator and discriminator.

### 4b. **DCGAN (2015)**

* **Key innovation:** Deep convolutional architectures for GANs.
* **Learning focus:** Conv layers, feature maps, improved stability.
* **Paper:** [DCGAN](https://arxiv.org/abs/1511.06434)
* **Datasets:** CIFAR-10, CelebA
* **Difficulty:** moderate
* **Project tasks:** implement convolutional generator/discriminator.

### 4c. **WGAN (2017)**

* **Key innovation:** Wasserstein distance for better loss stability.
* **Learning focus:** Earth-Mover distance, gradient clipping.
* **Paper:** [WGAN](https://arxiv.org/abs/1701.07875)
* **Datasets:** CIFAR-10, CelebA
* **Difficulty:** moderate
* **Project tasks:** implement WGAN loss, gradient penalty optional.

### 4d. **SAGAN (2018)**

* **Key innovation:** Self-attention in GANs for long-range dependencies.
* **Learning focus:** Attention layers in generator/discriminator.
* **Paper:** [SAGAN](https://arxiv.org/abs/1805.08318)
* **Datasets:** ImageNet subsets
* **Difficulty:** hard
* **Project tasks:** implement self-attention layers inside GAN.

---

## 5. **VQ & Discrete Latents**

### 5a. **VQ-VAE (2017)**

* **Key innovation:** Discrete latent codebooks.
* **Learning focus:** Vector quantization, straight-through estimator.
* **Paper:** [VQ-VAE](https://arxiv.org/abs/1711.00937)
* **Datasets:** CelebA, ImageNet subsets
* **Difficulty:** moderate → hard
* **Project tasks:** implement codebook, commitment/reconstruction loss.

### 5b. **VQ-VAE-2 (2019)**

* **Key innovation:** Multi-scale latent hierarchy.
* **Learning focus:** Multi-scale token modeling.
* **Paper:** [VQ-VAE-2](https://papers.neurips.cc/paper/9625-generating-diverse-high-fidelity-images-with-vq-vae-2.pdf)
* **Datasets:** CelebA, ImageNet
* **Difficulty:** hard
* **Project tasks:** implement hierarchical VQ-VAE-2, train on multi-scale latents.

---

## 6. **Transformer Decoders**

### 6a. **GPT-Style Transformer (2017)**

* **Key innovation:** Autoregressive self-attention.
* **Learning focus:** Masked self-attention, positional encodings.
* **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* **Datasets:** text corpora, VQ-token sequences
* **Difficulty:** moderate → hard
* **Project tasks:** implement small Transformer decoder for sequences.

---

## 7. **Diffusion Models**

### 7a. **DDPM (2020)**

* **Key innovation:** Reverse denoising process.
* **Learning focus:** Noise schedules, iterative sampling.
* **Paper:** [DDPM](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
* **Datasets:** CIFAR-10, CelebA
* **Difficulty:** moderate → hard
* **Project tasks:** train noise-predicting network, implement DDIM sampling.

### 7b. **Latent Diffusion (2022)**

* **Key innovation:** Diffusion in compressed latent space, CLIP guidance.
* **Learning focus:** Conditional generation, efficiency.
* **Paper:** [LDM](https://proceedings.mlr.press/v162/cohen22b/cohen22b.pdf)
* **Datasets:** LAION, CIFAR
* **Difficulty:** hard
* **Project tasks:** implement latent diffusion on CIFAR latents, add CLIP guidance.

---

## 8. **VQ + Perceptual Losses**

### 8a. **VQGAN (2019)**

* **Key innovation:** GAN + VQ + perceptual loss.
* **Learning focus:** Hybrid training losses, codebook interaction.
* **Paper:** [VQGAN](https://arxiv.org/abs/1906.00446)
* **Datasets:** CelebA, CIFAR
* **Difficulty:** hard
* **Project tasks:** implement reconstruction + perceptual + adversarial loss combination.

---

## 9. **Diffusion Transformers**

### 9a. **DiT (2022)**

* **Key innovation:** Transformer backbone for diffusion.
* **Learning focus:** Patch embeddings, residual attention.
* **Paper:** [DiT](https://arxiv.org/pdf/2501.01423.pdf)
* **Datasets:** ImageNet 256×256, LSUN
* **Difficulty:** hard
* **Project tasks:** implement patch embeddings, attention blocks, train on low-res ImageNet.

---

## 10. **Consistency & Rectified Flow**

### 10a. **Consistency Models (2023)**

* **Key innovation:** Deterministic sampling with consistency loss.
* **Learning focus:** ODE flows, fast sampling.
* **Paper:** [CS231n Lecture 14](https://cs231n.stanford.edu/slides/2025/lecture_14.pdf)
* **Datasets:** CIFAR-10, ImageNet
* **Difficulty:** moderate → hard
* **Project tasks:** implement ODE-based rectified flow sampler.

---

## 11. **3D / Neural Radiance Generative Models**

### 11a. **DreamFusion (2022)**

* **Key innovation:** Text-to-3D generation via differentiable rendering.
* **Paper:** [DreamFusion](https://www.sciencedirect.com/science/article/pii/S209526352400147X)
* **Difficulty:** hard
* **Project tasks:** implement 3D NeRF generator guided by diffusion prior.

---

## 12. **Foundation-Model-Aligned Latents**

### 12a. **VA-VAE + LightningDiT (2025)**

* **Key innovation:** Align latent space with pretrained vision foundation models.
* **Paper:** [VA-VAE + LightningDiT](https://arxiv.org/pdf/2501.01423.pdf)
* **Difficulty:** hard
* **Project tasks:** retrain VAE with alignment loss for better latent diffusion.

---

## 13. **Multimodal / LLM-Guided Generation**

### 13a. **Text-to-Image w/ LLM Guidance (2024–2025)**

* **Key innovation:** Merge LLMs with diffusion for semantic control.
* **Paper / Demo:** [FLUX](https://www.youtube.com/watch?v=Dmm4UG-6jxA)
* **Difficulty:** hard
* **Project tasks:** build lightweight text-to-image pipeline with CLIP embeddings conditioning diffusion.

---

