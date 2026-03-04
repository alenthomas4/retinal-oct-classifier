# OCT Retinal Classifier

Transfer learning pipeline for retinal OCT image analysis using a frozen ResNet50 backbone. Trained on the OLIVES dataset to perform dual-task classification: detecting 16 retinal biomarkers and classifying diabetic retinopathy disease labels. Embeddings are indexed with FAISS for similarity-based retrieval.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1o5AOY_Wc4yNnZsIy73GcABVVDJ6RkFoO?usp=sharing)

## Tech Stack
Python · PyTorch · ResNet50 · FAISS · LangChain · Google Gemini · HuggingFace Datasets

## What it does
- Loads pre-trained ResNet50, freezes the backbone, and attaches dual task-specific heads
- Biomarker head: multi-label classification across 16 retinal biomarkers
- Disease head: binary classification of diabetic retinopathy
- Extracts 2048-dim embeddings and indexes them with FAISS for retrieval
- Supports RAG-style querying using LangChain + Google Gemini

## Running in Colab

### Prerequisites
- A Google account with Google Drive access
- The OLIVES dataset parquet files saved to your Drive

### Steps

1. **Open the notebook** by clicking the badge above or going to:
   `https://colab.research.google.com/drive/1o5AOY_Wc4yNnZsIy73GcABVVDJ6RkFoO`

2. **Mount your Google Drive** by running the first cell:
```python
