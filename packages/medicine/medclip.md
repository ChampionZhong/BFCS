# medclip

[🔙 Back to Main Repo](../../../README.md) | [🔗 Original Repo](https://github.com/RyanWangZf/MedCLIP)

![Tool Count](https://img.shields.io/badge/Agent_Tools-3-blue?style=flat-square)
![Category](https://img.shields.io/badge/Category-Medicine-green?style=flat-square)
![Status](https://img.shields.io/badge/Import_Test-Passed-success?style=flat-square)

## 📖 Overview

MedCLIP is a PyTorch-based CLIP-style library that provides pretrained contrastive models and utilities to embed and match medical images and clinical text, enabling tasks like zero-shot/prompt-based medical image classification.

> **Note**: This documentation lists the **agent-ready wrapper functions** generated for this package. These functions have been strictly typed, docstring-enhanced, and tested for import stability within a standardized Apptainer environment.

## 🛠️ Available Agent Tools

Below is the list of **3** functions optimized for LLM tool-use.

| **Tool Name (Wrapper)**   | **Source**          | **File Path**     | **Arguments (Type)**        | **Description**                |
| ------------------------- | ------------------- | ----------------- | --------------------------- | ------------------------------ |
| `medclip_prompts_generate_chexpert_class_prompts` | `medclip.prompts.generate_chexpert_class_prompts` | `medclip/prompts.py` | `n: int = None` | `Generate text prompts for each CheXpert classification task used by MedCLIP for prompt-based (zero-shot or few-shot) image classification. This function builds candidate natural-language prompts for each CheXpert class by taking the Cartesian product of three token groups (severity, subtype, location) defined in constants.CHEXPERT_CLASS_PROMPTS for each class key. The resulting prompt strings (one token from each group joined with spaces) are intended to be fed into MedCLIP text encoders, processed by utilities such as medclip.prompts.process_class_prompts, and used by PromptClassifier for ensemble or zero-shot CheXpert label prediction as shown in the project README.` |
| `medclip_vision_model_window_partition` | `medclip.vision_model.window_partition` | `medclip/vision_model.py` | `x: torch.Tensor, window_size: int` | `medclip.vision_model.window_partition partitions a 4-D image feature tensor into non-overlapping square windows. This operation is used in MedCLIP vision models (for example ViT or Swin-style components) to prepare local image patches for windowed self-attention and downstream contrastive learning between medical images and text (e.g., chest x-ray feature maps). The function reshapes and permutes the input so each output row is one window, which simplifies batched window-wise processing used in MedCLIP training and inference.` |
| `medclip_vision_model_window_reverse` | `medclip.vision_model.window_reverse` | `medclip/vision_model.py` | `windows: torch.Tensor, window_size: int, H: int, W: int` | `medclip.vision_model.window_reverse reconstructs a full image tensor from non-overlapping square windows. This function is used in MedCLIP vision models to reverse a prior window partitioning step (for example, the window partition used in window-based Vision Transformer or shifted-window attention implementations). In the MedCLIP pipeline this reconstruction is necessary to restore spatial layout of per-window image features so they can be processed as a single image tensor for downstream embedding, contrastive alignment with text, or prompt-based classification.` |

## ⚖️ License

Original Code License: Unknown

Wrapper Code & Documentation: Apache-2.0

*This file was automatically generated on February 26, 2026.*
