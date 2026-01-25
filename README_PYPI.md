# 🦜 VieNeu-TTS

[![Awesome](https://img.shields.io/badge/Awesome-NLP-green?logo=github)](https://github.com/keon/awesome-nlp)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing)
[![Hugging Face 0.5B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-0.5B-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
[![Hugging Face 0.3B](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)

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

**Author:** Phạm Nguyễn Ngọc Bảo

---

## 📌 Table of Contents

1. [📦 Using the Python SDK](#sdk)
2. [🐳 Docker & Remote Server](#docker-remote)
3. [🎯 Custom Models](#custom-models)
4. [🛠️ Fine-tuning Guide](#finetuning)
5. [🔬 Model Overview](#backbones)
6. [🤝 Support & Contact](#support)

---

## 📦 1. Using the Python SDK (vieneu) <a name="sdk"></a>

Integrate VieNeu-TTS into your own software projects.

### Quick Install
```bash
# Windows (Avoid llama-cpp build errors)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# Linux / MacOS
pip install vieneu
```

### Quick Start (main.py)
```python
from vieneu import Vieneu
import os

# Initialization
tts = Vieneu()

# Standard synthesis (uses default voice)
text = "Xin chào, tôi là VieNeu. Tôi có thể giúp bạn đọc sách, làm chatbot thời gian thực, hoặc thậm chí clone giọng nói của bạn."
audio = tts.infer(text=text)
tts.save(audio, "standard_output.wav")
print("💾 Saved synthesis to: standard_output.wav")
```

*For full implementation details, see [main.py](main.py).*

---

## 🐳 2. Docker & Remote Server <a name="docker-remote"></a>

Deploy VieNeu-TTS as a high-performance API Server (powered by LMDeploy) with a single command.

### 1. Run with Docker (Recommended)

**Requirement**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is required for GPU support.

**Start the Server with a Public Tunnel (No port forwarding needed):**
```bash
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```

*   **Default**: The server loads the `VieNeu-TTS` model for maximum quality.
*   **Tunneling**: The Docker image includes a built-in `bore` tunnel. Check the container logs to find your public address (e.g., `bore.pub:31631`).

### 2. Using the SDK (Remote Mode)

Once the server is running, you can connect from anywhere (Colab, Web Apps, etc.) without loading heavy models locally:

```python
from vieneu import Vieneu
import os

# Configuration
REMOTE_API_BASE = 'http://your-server-ip:23333/v1'  # Or bore tunnel URL
REMOTE_MODEL_ID = "pnnbao-ump/VieNeu-TTS"

# Initialization (LIGHTWEIGHT - only loads small codec locally)
tts = Vieneu(mode='remote', api_base=REMOTE_API_BASE, model_name=REMOTE_MODEL_ID)
os.makedirs("outputs", exist_ok=True)

# List remote voices
available_voices = tts.list_preset_voices()
for desc, name in available_voices:
    print(f"   - {desc} (ID: {name})")

# Use specific voice (dynamically select second voice)
if available_voices:
    _, my_voice_id = available_voices[1]
    voice_data = tts.get_preset_voice(my_voice_id)
    audio_spec = tts.infer(text="Chào bạn, tôi đang nói bằng giọng của bác sĩ Tuyên.", voice=voice_data)
    tts.save(audio_spec, f"outputs/remote_{my_voice_id}.wav")
    print(f"💾 Saved synthesis to: outputs/remote_{my_voice_id}.wav")

# Standard synthesis (uses default voice)
text_input = "Chế độ remote giúp tích hợp VieNeu vào ứng dụng Web hoặc App cực nhanh mà không cần GPU tại máy khách."
audio = tts.infer(text=text_input)
tts.save(audio, "outputs/remote_output.wav")
print("💾 Saved remote synthesis to: outputs/remote_output.wav")

# Zero-shot voice cloning (encodes audio locally, sends codes to server)
if os.path.exists("examples/audio_ref/example_ngoc_huyen.wav"):
    cloned_audio = tts.infer(
        text="Đây là giọng nói được clone và xử lý thông qua VieNeu Server.",
        ref_audio="examples/audio_ref/example_ngoc_huyen.wav",
        ref_text="Tác phẩm dự thi bảo đảm tính khoa học, tính đảng, tính chiến đấu, tính định hướng."
    )
    tts.save(cloned_audio, "outputs/remote_cloned_output.wav")
    print("💾 Saved remote cloned voice to: outputs/remote_cloned_output.wav")
```

*For full implementation details, see: [main_remote.py](main_remote.py)*

### Voice Preset Specification (v1.0)
VieNeu-TTS uses the official `vieneu.voice.presets` specification to define reusable voice assets.
Only `voices.json` files following this spec are guaranteed to be compatible with VieNeu-TTS SDK ≥ v1.x.

### 3. Advanced Configuration

Customize the server to run specific versions or your own fine-tuned models.

**Run the 0.3B Model (Faster):**
```bash
docker run --gpus all pnnbao/vieneu-tts:serve --model pnnbao-ump/VieNeu-TTS-0.3B --tunnel
```

**Serve a Local Fine-tuned Model:**
If you have merged a LoRA adapter, mount your output directory to the container:
```bash
# Linux / macOS
docker run --gpus all \
  -v $(pwd)/finetune/output:/workspace/models \
  pnnbao/vieneu-tts:serve \
  --model /workspace/models/merged_model --tunnel
```

*For full implementation details, see: [main_remote.py](main_remote.py)*

---

## 🎯 3. Custom Models (LoRA, GGUF, Finetune) <a name="custom-models"></a>

VieNeu-TTS allows you to load custom models directly from HuggingFace or local paths via the Web UI.

*👉 See the detailed guide at: **[docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)***

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

## 📚 References

- **Dataset:** [VieNeu-TTS-1000h (Hugging Face)](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h)
- **Model 0.5B:** [pnnbao-ump/VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- **Model 0.3B:** [pnnbao-ump/VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
- **LoRA Guide:** [docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)

---

## 🤝 6. Support & Contact <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Join our community](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/bao.phamnguyenngoc.5)
- **Licensing:** 
  - **VieNeu-TTS (0.5B):** Apache 2.0 (Free to use).
  - **VieNeu-TTS-0.3B:** CC BY-NC 4.0 (Non-commercial).
    - ✅ **Free:** For students, researchers, and non-profit purposes.
    - ⚠️ **Commercial/Enterprise:** Contact the author for licensing.

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
