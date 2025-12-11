# Vietnamese OCR System: CTC vs Seq2Seq Comparison 🇻🇳

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/vinhisreal/ocr-vietnam-invoice-ocr)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

An end-to-end Vietnamese Optical Character Recognition (OCR) application. This project integrates Text Detection and provides a comparative analysis between two recognition architectures: **CTC (Connectionist Temporal Classification)** and **Seq2Seq (Sequence-to-Sequence)**.

Built with **PyTorch** and deployed with an interactive **Gradio** web interface.

🔗 **[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/vinhisreal/ocr-vietnam-invoice-ocr)**

## ✨ Key Features

* **Text Detection:** Utilizes **YOLOv8** (with fallback mechanisms to YOLOv5 or OpenCV EAST) to accurately localize text lines in receipts and documents.
* **Text Recognition:** A side-by-side comparison of two Deep Learning approaches:
    * **CTC Model:** CNN + Transformer Encoder + CTC Loss.
    * **Seq2Seq Model:** CNN + Transformer Encoder + Transformer Decoder (with Attention mechanism).
* **Interactive UI:** Upload images, adjust confidence thresholds, filter text regions by area, and inspect individual cropped results.
* **Vietnamese Support:** Custom tokenizer specifically designed to handle full Vietnamese diacritics and special characters.

## 🧠 Model Architecture

### 1. Text Detection
The system primarily uses **YOLOv8** to detect bounding boxes for text lines. It includes a robust fallback system: if YOLOv8 fails to load, it attempts to use YOLOv5, and finally falls back to the OpenCV EAST text detector.

### 2. Text Recognition
* **Feature Extraction:** Both models share a common **CNN backbone** (ResNet-like layers) to extract visual features from the cropped text regions.
* **CTC Model:** Visual features are passed through a Transformer Encoder. The output is decoded using Greedy Search with CTC Loss. This model is generally faster and suitable for simpler text structures.
* **Seq2Seq Model:** utilizes a full Transformer **Encoder-Decoder** architecture. The Decoder uses an attention mechanism to predict the next character based on visual features and the previous token. This usually yields higher accuracy for complex text but is computationally heavier than CTC.

## 🛠️ Local Installation

To run this project locally with all model weights included, follow these steps:

1. Clone the repository
Clone the project directly from Hugging Face:

```Bash

git clone [https://huggingface.co/spaces/vinhisreal/ocr-vietnam-invoice-ocr](https://huggingface.co/spaces/vinhisreal/ocr-vietnam-invoice-ocr)
cd ocr-vietnam-invoice-ocr
```
2. Install dependencies
Python 3.10+ is recommended.
```Bash
pip install -r requirements.txt
```
Note: The Model/ folder containing yolov8_best.pt, ctcmodel.pth, and tranformerdecoder.pth is included in the clone.

## 🚀 Usage
Run the Gradio app:

```Bash
python app.py
```
Open your browser at the local URL provided (usually http://127.0.0.1:7860).
Upload: Select an image of a Vietnamese receipt or document.

## Configure:

Confidence Threshold: Lower values detect more text but may include noise.
Min/Max Area: Filter out noise or large non-text blocks.
Process: Click "Xử lý ảnh" (Process Image).
Results: View the detected boxes, cropped regions, and the comparison table between CTC and Seq2Seq outputs.

## 📄 License
MIT License