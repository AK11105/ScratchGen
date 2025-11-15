# 📚 Generative Models — Learning Timeline: From-Scratch PyTorch Implementations

---

## 1. **Variational Autoencoder (VAE, 2013)** — **IMPLEMENT**

* **Key innovation:** Reparameterization trick, ELBO training.
* **Learning focus:** Probabilistic encoders/decoders, KL divergence, latent traversals.
* **Paper:** Auto-Encoding Variational Bayes
* **Datasets:** MNIST → FashionMNIST → CIFAR-10
* **Difficulty:** easy → moderate
* **Project tasks:** implement Gaussian encoder/decoder, ELBO, latent interpolation.

---

## 1a. **Denoising Autoencoder (DAE, 2010)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Reconstruction from corrupted inputs.
* **Learning focus:** Noise robustness.
* **Paper:** Denoising Autoencoders
* **Datasets:** MNIST
* **Difficulty:** easy
* **Project tasks:** add noise + reconstruct clean targets.

---

## 1b. **Sparse & Contractive Autoencoders (2011–2012)** — **READ ONLY**

* **Key innovation:** Sparsity / Jacobian penalties.
* **Learning focus:** Representation regularization.
* **Paper:** Sparse AEs, Contractive AEs
* **Datasets:** MNIST
* **Difficulty:** easy
* **Project tasks:** optional α-L1 or contractive loss experiments.

---

## 2. **Autoregressive Models**

### 2a. **PixelRNN (2016)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Pixel-by-pixel sequential prediction.
* **Learning focus:** RNN factorization of images.
* **Paper:** PixelRNN
* **Datasets:** MNIST, CIFAR-10
* **Difficulty:** moderate
* **Project tasks:** implement masked RNN sampler.

### 2b. **PixelCNN (2016)** — **IMPLEMENT**

* **Key innovation:** Masked convolutions.
* **Learning focus:** Parallel autoregressive modeling.
* **Paper:** PixelCNN
* **Datasets:** MNIST, CIFAR-10
* **Difficulty:** moderate
* **Project tasks:** build masked conv layers + sampling.

### 2c. **PixelCNN++ (2017)** — **READ ONLY / OPTIONAL PARTIAL**

* **Key innovation:** Discretized mixture logistics.
* **Learning focus:** Improved pixel likelihoods.
* **Paper:** PixelCNN++
* **Datasets:** CIFAR-10
* **Difficulty:** moderate
* **Project tasks:** optional mixture logistic head.

---

## 3. **Flow-Based Models**

### 3a. **RealNVP (2016)** — **IMPLEMENT**

* **Key innovation:** Invertible affine coupling.
* **Learning focus:** Jacobians, exact likelihood.
* **Paper:** RealNVP
* **Datasets:** CIFAR-10
* **Difficulty:** moderate → hard
* **Project tasks:** build coupling layers, compute log-dets.

### 3b. **Glow (2018)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Invertible 1×1 conv, actnorm.
* **Learning focus:** scalable flows.
* **Paper:** Glow
* **Datasets:** CIFAR-10, ImageNet
* **Difficulty:** hard
* **Project tasks:** implement 1×1 conv + actnorm; skip full multi-scale if too heavy.

### 3c. **MAF / MADE (2015–2017)** — **READ ONLY**

* **Key innovation:** Autoregressive flows.
* **Learning focus:** masked linear models.
* **Papers:** MADE, MAF
* **Difficulty:** moderate
* **Project tasks:** optional small MADE layer experiment.

---

## 4. **Energy-Based & Score-Matching Models**

### 4a. **Energy-Based Models (EBMs)** — **READ ONLY**

* **Key innovation:** Unnormalized energy functions.
* **Learning focus:** MCMC, contrastive divergence.
* **Paper:** EBM surveys
* **Datasets:** small toy images
* **Difficulty:** hard
* **Project tasks:** optional Langevin sampler.

### 4b. **Score-Based Generative Models (2019–2020)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Learning ∇x log p(x).
* **Learning focus:** score matching + SDE samplers.
* **Paper:** Score-Based Generative Modeling
* **Datasets:** CIFAR-10
* **Difficulty:** hard
* **Project tasks:** implement denoising score matching + simple Langevin sampler.

---

## 5. **Generative Adversarial Networks (GANs)**

### 5a. **Vanilla GAN (2014)** — **IMPLEMENT**

* **Key innovation:** Adversarial min-max.
* **Datasets:** MNIST
* **Difficulty:** moderate
* **Project tasks:** implement generator + discriminator.

### 5b. **DCGAN (2015)** — **IMPLEMENT**

* **Key innovation:** Convolutional GANs.
* **Datasets:** CelebA
* **Difficulty:** moderate
* **Project tasks:** implement full DCGAN.

### 5c. **WGAN & WGAN-GP (2017)** — **IMPLEMENT**

* **Key innovation:** Wasserstein distance + gradient penalty.
* **Datasets:** CIFAR-10, CelebA
* **Difficulty:** moderate
* **Project tasks:** implement critic, GP, improved stability.

### 5d. **SAGAN (2018)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Self-attention in GANs.
* **Datasets:** ImageNet subsets
* **Difficulty:** hard
* **Project tasks:** integrate attention block into DCGAN/WGAN.

---

## 6. **VQ & Discrete Latents**

### 6a. **VQ-VAE (2017)** — **IMPLEMENT**

* **Key innovation:** Discrete codebooks, straight-through estimator.
* **Datasets:** CelebA
* **Difficulty:** moderate → hard
* **Project tasks:** codebook, commitment loss.

### 6b. **VQ-VAE-2 (2019)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Multi-level discrete latents.
* **Datasets:** ImageNet
* **Difficulty:** hard
* **Project tasks:** two-level VQ only (full model is heavy).

---

## 7. **Disentanglement Models**

### 7a. **β-VAE (2017)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** KL-scaled disentanglement.
* **Datasets:** dSprites
* **Difficulty:** easy
* **Project tasks:** modify VAE loss, visualize factors.

### 7b. **FactorVAE (2018)** — **READ ONLY**

* **Key innovation:** Total correlation penalty.
* **Datasets:** dSprites
* **Difficulty:** moderate
* **Project tasks:** optional TC discriminator.

---

## 8. **Transformer Decoders**

### 8a. **GPT-Style Transformer (2017)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Autoregressive self-attention.
* **Datasets:** text or VQ tokens
* **Difficulty:** moderate → hard
* **Project tasks:** small decoder-only transformer.

---

## 9. **Diffusion Models**

### 9a. **DDPM (2020)** — **IMPLEMENT**

* **Key innovation:** Reverse diffusion.
* **Datasets:** CIFAR-10
* **Difficulty:** moderate → hard
* **Project tasks:** full DDPM training + DDIM sampling.

### 9b. **Score-SDE (2019–2020)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** continuous-time formulation.
* **Datasets:** CIFAR-10
* **Difficulty:** hard
* **Project tasks:** convert DDPM U-Net into SDE sampler.

### 9c. **Latent Diffusion (2022)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** Diffusion in latent VAE space.
* **Datasets:** CIFAR (toy), LAION (full)
* **Difficulty:** hard
* **Project tasks:** tiny latent diffusion setup.

### 9d. **Classifier-Free Guidance & Cross-Attention (2022)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** conditioning + guidance scale.
* **Datasets:** image–text pairs
* **Difficulty:** moderate
* **Project tasks:** add text embeddings and cross-attention blocks.

---

## 10. **VQ + Perceptual Losses**

### 10a. **VQGAN (2019)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** perceptual + adversarial loss + VQ.
* **Datasets:** CelebA
* **Difficulty:** hard
* **Project tasks:** simplified VQGAN (skip full discriminator pyramid).

---

## 11. **Diffusion Transformers**

### 11a. **DiT (2022)** — **READ ONLY / OPTIONAL PARTIAL**

* **Key innovation:** Transformers as diffusion denoisers.
* **Datasets:** ImageNet
* **Difficulty:** hard
* **Project tasks:** implement minimal DiT block.

---

## 12. **Consistency & Rectified Flow**

### 12a. **Consistency Models (2023)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** deterministic fast samplers.
* **Datasets:** CIFAR-10
* **Difficulty:** hard
* **Project tasks:** implement consistency loss + ODE sampler.

---

## 13. **3D / Neural Radiance Generative Models**

### 13a. **DreamFusion (2022)** — **READ ONLY**

* **Key innovation:** diffusion-guided NeRF optimization.
* **Datasets:** rendered small scenes
* **Difficulty:** very hard
* **Project tasks:** optional tiny NeRF + image prior.

---

## 14. **Foundation-Model-Aligned Latents**

### 14a. **VA-VAE + LightningDiT (2025)** — **READ ONLY**

* **Key innovation:** align VAE latents with foundation vision models.
* **Difficulty:** hard
* **Project tasks:** none required.

---

## 15. **Multimodal / LLM-Guided Generation**

### 15a. **Text-to-Image w/ LLM Guidance (2024–2025)** — **PARTIALLY IMPLEMENT**

* **Key innovation:** semantic conditioning from LLM outputs.
* **Datasets:** image–text pairs
* **Difficulty:** hard
* **Project tasks:** LLM → CLIP embed → latent diffusion conditioning.

---

