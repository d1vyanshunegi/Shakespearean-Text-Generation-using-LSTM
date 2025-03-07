# Shakespearean-Text-Generation-using-LSTM

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A production-ready LSTM implementation for character-level text generation**, trained on Shakespearean works. This PyTorch-based project demonstrates sequence modeling while generating literary-quality text.

## Overview
This repository contains an end-to-end solution for:
- Character-level text generation using 2-layer LSTM
- Training pipeline with validation
- Temperature-controlled sampling
- Model checkpointing & GPU acceleration

Trained on 40,000+ lines of Shakespearean text, the model learns to generate surprisingly coherent pseudo-Elizabethan prose/poetry.

## Features
- **Industrial Engineering Practices**
  - Gradient clipping & learning rate scheduling
  - Batch processing with DataLoader
  - Validation loss tracking
  - Automatic device detection (CUDA/CPU)
- **Model Architecture**
  - Embedding layer + LSTM + Dropout
  - Character-level vocabulary
  - Hidden state preservation
- **Text Generation**
  - Temperature-based sampling
  - Seed sequence initialization
  - Context window sliding
