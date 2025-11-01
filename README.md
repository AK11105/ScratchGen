# 🎨 ScratchGen — Generative Modeling from Scratch (PyTorch)

> *“Building every generative architecture by hand — to understand, not just to use.”*

---

## 🧩 Overview

**ScratchGen** is a hands-on project for learning and implementing **modern generative models from scratch** using **PyTorch**, organized by historical and conceptual breakthroughs.

Each stage focuses on **rebuilding key generative architectures** — from *VAEs and GANs* to *Diffusion Transformers and Multimodal Generators* — directly from foundational papers.

---


## 🗂 Folder Structure

```text
notebooks/      # Jupyter notebooks for guided experiments
data/           # Raw and processed datasets
src/            # Core modules: models, trainers, evaluators, utils
experiments/    # Saved checkpoints, logs, and results
scripts/        # Run scripts for training and evaluation
tests/          # Unit tests for project modules
````

---

## 📖 Learning Roadmap

Implementation order, references, datasets, and difficulty ratings are all documented in  
👉 **[`TIMELINE.md`](./TIMELINE.md)**

That file defines the **canonical progression** of ScratchGen — from probabilistic VAEs to modern multimodal diffusion systems.

---

## 🧠 Philosophy

> “Code is the curriculum.”

ScratchGen emphasizes **re-derivation and self-implementation**:
- No prebuilt architectures.  
- Each model trained end-to-end.  
- Minimal dependencies beyond PyTorch and standard math.  
- Strong focus on **intuition → derivation → reproducible results**.

---

## 🚀 Goals

- Build intuition for each class of generative model.  
- Understand the *why* behind each innovation.  
- Create a modular foundation to experiment across architectures.  
- Document each implementation with clean notebooks, visuals, and equations.

---

## 🧰 Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- Matplotlib, NumPy, tqdm, torchvision  
- Optional: CUDA, Weights & Biases (for logging)

Install dependencies:
```bash
pip install -r requirements.txt
````

---

## 🔗 References

* Full list of key papers and implementation sequence in **[`TIMELINE.md`](./TIMELINE.md)**
* Base inspiration: *Kingma & Welling (2013)* → *Goodfellow (2014)* → *Ho et al. (2020)* → *Peebles & Xie (2022)* → *Yao et al. (2025)*

---

> 📘 **Start here:** open [`TIMELINE.md`](./TIMELINE.md) to follow the exact implementation order.
