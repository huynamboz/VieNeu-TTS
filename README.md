# 🦜 VieNeu-TTS 

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.5B-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B--GGUF-green)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing) 

<img width="899" height="615" alt="VieNeu-TTS UI" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS** is an advanced on-device Vietnamese Text-to-Speech (TTS) model with **instant voice cloning**.

> [!TIP]
> **Voice Cloning:** All model variants (including GGUF) support instant voice cloning with just **3-5 seconds** of reference audio.

This project features two core architectures trained on the [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) dataset:
- **VieNeu-TTS (0.5B):** An enhanced model fine-tuned from the NeuTTS Air architecture for maximum stability.
- **VieNeu-TTS-0.3B:** A specialized model **trained from scratch** using the VieNeu-TTS-1000h dataset, delivering 2x faster inference and ultra-low latency.

These represent a significant upgrade from the previous VieNeu-TTS-140h with the following improvements:
- **Enhanced pronunciation**: More accurate and stable Vietnamese pronunciation
- **Code-switching support**: Seamless transitions between Vietnamese and English
- **Better voice cloning**: Higher fidelity and speaker consistency
- **Real-time synthesis**: 24 kHz waveform generation on CPU or GPU
- **Multiple model formats**: Support for PyTorch, GGUF Q4/Q8 (CPU optimized), and ONNX codec

VieNeu-TTS delivers production-ready speech synthesis fully offline.
- **Author:** Phạm Nguyễn Ngọc Bảo

---

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 📌 Table of Contents

1. [🦜 Installation & Web UI](#installation)
2. [📦 Using the Python SDK](#sdk)
3. [🎯 Custom Models](#custom-models)
4. [🛠️ Fine-tuning Guide](#finetuning)
5. [🔬 Model Overview](#backbones)
6. [🐋 Deployment with Docker](#docker)
7. [🤝 Support & Contact](#support)

---

## 🦜 1. Installation & Web UI <a name="installation"></a>

The fastest way to experience VieNeu-TTS is through the Web interface (Gradio).

### System Requirements
- **Python:** 3.10 - 3.12 (3.12 recommended)
- **eSpeak NG:** Required for phonemization.
  - **Windows:** Download the `.msi` from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases).
  - **macOS:** `brew install espeak`
  - **Ubuntu/Debian:** `sudo apt install espeak-ng`
- **NVIDIA GPU (Optional):** For maximum speed via LMDeploy. Requires **CUDA 12.8 or newer** and [NVIDIA GPU Computing Toolkit](https://developer.nvidia.com/cuda-downloads) installed.

### Installation Steps
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Environment Setup with `uv` (Recommended):**
   - **Step A: Install uv (if you haven't)**
     ```bash
     # Windows:
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     
     # Linux/macOS:
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   - **Step B: Install dependencies**
     - **Option 1: Default (with GPU support)**
       ```bash
       uv sync
       ```
     - **Option 2: CPU-ONLY (Lightweight version)**
       ```bash
       uv sync --no-default-groups
       ```

3. **Start the Web UI:**
   ```bash
   uv run gradio_app.py
   ```
   Access the UI at `http://127.0.0.1:7860`.

---

## 📦 2. Using the Python SDK (vieneu) <a name="sdk"></a>

Integrate VieNeu-TTS into your own software projects.

### Quick Install
```bash
# Windows (Avoid llama-cpp build errors)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# Linux / MacOS
pip install vieneu
```

### Full Features Example (main.py)
```python
"""
Demo VieNeuSDK v1.1.3 - Full Features Guide
"""

import time
import soundfile as sf
from vieneu import Vieneu
from pathlib import Path

def main():
    print("🚀 Initializing VieNeu SDK (v1.1.3)...")
    
    # Initialize SDK
    # Default: "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf" (Speed & CPU Optimized)
    # Mode selection:
    # - mode="standard" (Default): Runs locally using GGUF (CPU) or PyTorch
    # - mode="remote": Connects to the LMDeploy server setup above for max speed
    
    tts = Vieneu()
    # Or to use Remote mode (Must start 'lmdeploy serve api_server pnnbao-ump/VieNeu-TTS-0.3B' first):
    # tts = Vieneu(model_name="pnnbao-ump/VieNeu-TTS-0.3B", mode="remote", api_base="http://localhost:23333/v1")

    # ---------------------------------------------------------
    # PART 1: PRESET VOICES
    # ---------------------------------------------------------
    print("\n--- 1. Available Preset Voices ---")
    available_voices = tts.list_preset_voices()
    print("📋 Voices:", available_voices)
    
    # Select a preset voice
    current_voice = tts.get_preset_voice("Binh")
    print("✅ Selected voice: Binh")


    # ---------------------------------------------------------
    # PART 2: CREATE & SAVE CUSTOM VOICE
    # ---------------------------------------------------------
    print("\n--- 2. Create Custom Voice ---")
    
    # Replace with your actual .wav file path and its exact transcript
    sample_audio = Path(__file__).parent / "example.wav"
    sample_text = "ví dụ 2. tính trung bình của dãy số."

    if sample_audio.exists():
        voice_name = "MyCustomVoice"
        print(f"🎙️ Cloning voice from: {sample_audio.name}")
        
        # 'clone_voice' supports saving directly with 'name' argument
        custom_voice = tts.clone_voice(
            audio_path=sample_audio,
            text=sample_text,
            name=voice_name  # <-- Automatically saves voice to system
        )
        print(f"✅ Voice created and saved as: '{voice_name}'")
        
        # Switch to new voice
        current_voice = custom_voice


    # ---------------------------------------------------------
    # PART 3: SYNTHESIS WITH ADVANCED PARAMETERS
    # ---------------------------------------------------------
    print("\n--- 3. Speech Synthesis ---")
    text_input = "Xin chào, tôi là VieNeu-TTS. Tôi có thể giúp bạn đọc sách, hoặc clone giọng nói của bạn."
    
    print("🎧 Generating...")
    audio = tts.infer(
        text=text_input,
        voice=current_voice,
        temperature=1.0,  # Adjustable: 0.1 -> Stable, 1.0+ -> Expressive
        top_k=50
    )
    sf.write("output.wav", audio, 24000)
    print("💾 Saved: output.wav")

    tts.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
```
*For more scripts, see [main.py](main.py) in the project root.*

---

## 🎯 3. Custom Models (LoRA, GGUF, Finetune) <a name="custom-models"></a>

VieNeu-TTS allows you to load custom models directly from HuggingFace or local paths via the Web UI.

- **LoRA Support:** Automatically merges LoRA into the base model and accelerates with **LMDeploy**.
- **GGUF Support:** Runs smoothly on CPU using the llama.cpp backend.
- **Private Repos:** Supports entering an HF Token to access private models.

👉 See the detailed guide at: **[docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)**

---

## 🛠️ 4. Fine-tuning Guide <a name="finetuning"></a>

Train VieNeu-TTS on your own voice or custom datasets.

- **Simple Workflow:** Use the `train.py` script with optimized LoRA configurations.
- **Documentation:** Follow the step-by-step guide in **[finetune/README.md](finetune/README.md)**.
- **Notebook:** Experience it directly on Google Colab via `finetune/finetune_VieNeu-TTS.ipynb`.

---

## 🔬 5. Model Overview (Backbones) <a name="backbones"></a>

| Model                   | Format  | Device  | Quality    | Speed                   |
| ----------------------- | ------- | ------- | ---------- | ----------------------- |
| VieNeu-TTS              | PyTorch | GPU/CPU | ⭐⭐⭐⭐⭐ | Very Fast with lmdeploy |
| VieNeu-TTS-0.3B         | PyTorch | GPU/CPU | ⭐⭐⭐⭐   | **Ultra Fast (2x)**     |
| VieNeu-TTS-q8-gguf      | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐   | Fast                    |
| VieNeu-TTS-q4-gguf      | GGUF Q4 | CPU/GPU | ⭐⭐⭐     | Very Fast               |
| VieNeu-TTS-0.3B-q8-gguf | GGUF Q8 | CPU/GPU | ⭐⭐⭐⭐   | **Ultra Fast (1.5x)**   |
| VieNeu-TTS-0.3B-q4-gguf | GGUF Q4 | CPU/GPU | ⭐⭐⭐     | **Extreme Speed (2x)**  |

### 🔬 Model Details

- **Training Data:** [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) — 443,641 curated Vietnamese samples (Used for all versions).
- **Audio Codec:** NeuCodec (Torch implementation; ONNX & quantized variants supported).
- **Context Window:** 2,048 tokens shared by prompt text and speech tokens.
- **Output Watermark:** Enabled by default.

---

## 🐋 6. Deployment with Docker <a name="docker"></a>

Deploy quickly without manual environment setup.

```bash
# Run with CPU
docker compose --profile cpu up

# Run with GPU (Requires NVIDIA Container Toolkit)
docker compose --profile gpu up
```
Check [docs/Deploy.md](docs/Deploy.md) for more details.

---

## 📚 References

- **Dataset:** [VieNeu-TTS-1000h (Hugging Face)](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h)
- **Model 0.5B:** [pnnbao-ump/VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- **Model 0.3B:** [pnnbao-ump/VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
- **LoRA Guide:** [docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)

---

## 🤝 7. Support & Contact <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/bao.phamnguyenngoc.5)
- **Licensing:** 
  - **VieNeu-TTS (0.5B):** Apache 2.0 (Free to use).
  - **VieNeu-TTS-0.3B:** CC BY-NC 4.0 (Non-commercial).
    - ✅ **Free:** For students, researchers, and non-profit purposes.
    - ⚠️ **Commercial/Enterprise:** Contact the author for licensing (Estimated: **5,000 USD/year** - negotiable).

---

## 📑 Citation

```bibtex
@misc{vieneutts2026,
  title        = {VieNeu-TTS: Vietnamese Text-to-Speech with Instant Voice Cloning},
  author       = {Pham Nguyen Ngoc Bao},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/pnnbao-ump/VieNeu-TTS}}
}
```

---

## 🙏 Acknowledgements

This project builds upon the [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) and [NeuCodec](https://huggingface.co/neuphonic/neucodec) architectures. Specifically, the **VieNeu-TTS (0.5B)** model is fine-tuned from NeuTTS Air, while the **VieNeu-TTS-0.3B** model is a custom architecture trained from scratch using the [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) dataset.

---

**Made with ❤️ for the Vietnamese TTS community**
