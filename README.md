# 🖼️ Image Caption Generator

A Python-based image captioning application that automatically generates natural language descriptions for images. The project supports **multiple state-of-the-art pre-trained models** and provides **bilingual output in both English (EN) and Turkish (TR)**.

---

## 🚀 About

This project leverages several modern vision-language models to generate descriptive captions from images — no custom training required. Users can select from multiple models depending on their needs (speed, accuracy, language support) and receive captions in English or Turkish. The Turkish output is produced via translation from English captions, enabling native-language accessibility.

---

## 🤖 Supported Models

| Model | Description |
|-------|-------------|
| **BLIP** | Bootstrapping Language-Image Pre-training — robust general-purpose captioning |
| **CLIP** | Contrastive Language-Image Pre-training by OpenAI — powerful image-text alignment |
| **GIT Caption** | Generative Image-to-text Transformer by Microsoft |
| **ViT-GPT2 Caption** | Vision Transformer (ViT) encoder + GPT-2 decoder for caption generation |
| **LLaVA** | Large Language and Vision Assistant — instruction-following multimodal model |

All models are pre-trained and loaded directly — **no dataset or model training required**.

---

## 🌐 Language Support

- 🇬🇧 **English (EN)** — native output from all models
- 🇹🇷 **Turkish (TR)** — translated output for Turkish language accessibility

---

## ✨ Features

- 📸 Generate captions from any input image
- 🤖 Choose from 5 different pre-trained vision-language models
- 🌐 Bilingual caption output (English & Turkish)
- ⚡ No training or dataset needed — plug and play with pre-trained weights
- 🔁 Compare outputs across multiple models for the same image

---

## 🛠️ Tech Stack

| Technology | Description |
|------------|-------------|
| **Python** | Core programming language (100%) |
| **Hugging Face Transformers** | Model loading and inference (BLIP, GIT, ViT-GPT2, LLaVA) |
| **CLIP (OpenAI)** | Image-text contrastive model |
| **Pillow** | Image loading and preprocessing |
| **Deep Translator / Googletrans** | English → Turkish translation |
| **Torch** | Deep learning backend |

---

## 🔄 How It Works

```
Input Image
     │
     ▼
Image Preprocessing (resize, normalize)
     │
     ▼
Selected Model (BLIP / CLIP / GIT / ViT-GPT2 / LLaVA)
     │
     ▼
English Caption Generated
     │
     ▼
Translation Layer (EN → TR)
     │
     ▼
Output: Caption in English & Turkish
```

---

## 📁 Project Structure

```
image-caption-generator/
├── resimden_aciklama/     # Main source code (model loading, captioning, translation)
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.8+
- pip
- GPU recommended (CUDA) for faster inference with larger models like LLaVA

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BerraCevik/image-caption-generator.git
   cd image-caption-generator
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install transformers torch pillow deep-translator openai-clip
   ```

4. **Run the caption generator:**
   ```bash
   cd resimden_aciklama
   python main.py
   ```

---

## 💡 Example Usage

```python
# Generate caption for an image using BLIP model
generate_caption(image_path="photo.jpg", model="blip", lang="tr")

# Output:
# EN: "a dog playing in the park"
# TR: "parkta oynayan bir köpek"
```

---

## 🧠 Model Comparison

| Model | Speed | Accuracy | Turkish Support |
|-------|-------|----------|-----------------|
| BLIP | ⚡⚡⚡ | ⭐⭐⭐⭐ | ✅ (via translation) |
| CLIP | ⚡⚡⚡ | ⭐⭐⭐ | ✅ (via translation) |
| GIT Caption | ⚡⚡ | ⭐⭐⭐⭐ | ✅ (via translation) |
| ViT-GPT2 | ⚡⚡ | ⭐⭐⭐ | ✅ (via translation) |
| LLaVA | ⚡ | ⭐⭐⭐⭐⭐ | ✅ (via translation) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is open source and developed for educational purposes.

---
