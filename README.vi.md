# 🦜 VieNeu-TTS

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.5B-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B--GGUF-green)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing) 

<img width="899" height="615" alt="VieNeu-TTS UI" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS** là mô hình Text-to-Speech (TTS) tiếng Việt tiên tiến hỗ trợ **Instant Voice Cloning** (tái tạo giọng nói tức thì) chỉ với 3-5 giây âm thanh mẫu.
- **Author:** Phạm Nguyễn Ngọc Bảo

---

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 📌 Mục lục

1. [🦜 Cài đặt & Chạy Web UI](#cài-đặt)
2. [📦 Sử dụng Python SDK](#sdk)
3. [🎯 Custom Model](#custom-model)
4. [🛠️ Hướng dẫn Fine-tuning](#finetuning)
5. [🔬 Tổng quan mô hình](#backbones)
6. [🐋 Triển khai với Docker](#docker)
7. [🤝 Hỗ trợ & Liên hệ](#hỗ-trợ)

---

## 🦜 1. Cài đặt & Chạy Web UI <a name="cài-đặt"></a>

Cách nhanh nhất để trải nghiệm VieNeu-TTS là sử dụng giao diện Web (Gradio).

### Yêu cầu hệ thống
- **Python:** 3.10 - 3.12 (Khuyên dùng 3.12)
- **eSpeak NG:** Bắt buộc để xử lý phiên âm.
  - **Windows:** Tải `.msi` từ [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases).
  - **macOS:** `brew install espeak`
  - **Ubuntu/Debian:** `sudo apt install espeak-ng`
- **NVIDIA GPU (Tùy chọn):** Để đạt tốc độ tối đa với LMDeploy. Yêu cầu **CUDA 12.8 trở lên** và cài đặt [NVIDIA GPU Computing Toolkit](https://developer.nvidia.com/cuda-downloads).

### Các bước cài đặt
1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Cài đặt môi trường với `uv` (Khuyên dùng):**
   - **Bước A: Cài đặt uv (nếu chưa có)**
     ```bash
     # Windows:
     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
     
     # Linux/macOS:
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```

   - **Bước B: Cài đặt dependencies**
     - **Cách 1: Mặc định (có hỗ trợ GPU)**
       ```bash
       uv sync
       ```
     - **Cách 2: Chỉ dùng CPU (Bản rút gọn)**
       ```bash
       uv sync --no-default-groups
       ```

3. **Chạy giao diện Web:**
   ```bash
   uv run gradio_app.py
   ```
   Truy cập `http://127.0.0.1:7860` để bắt đầu.

---

## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

Nếu bạn muốn tích hợp VieNeu-TTS vào dự án phần mềm của mình.

### Cài đặt nhanh
```bash
# Windows (Tránh lỗi build llama-cpp)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# Linux / MacOS
pip install vieneu
```

### Hướng dẫn sử dụng đầy đủ (main.py)
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
    # Mặc định: "pnnbao-ump/VieNeu-TTS-0.3B-q4-gguf" (Tối ưu cho CPU)
    # Chế độ:
    # - mode="standard" (Mặc định): Chạy local
    # - mode="remote": Kết nối tới LMDeploy server
    
    tts = Vieneu()
    # Hoặc dùng Remote mode:
    # tts = Vieneu(model_name="pnnbao-ump/VieNeu-TTS-0.3B", mode="remote", api_base="http://localhost:23333/v1")

    # ---------------------------------------------------------
    # PHẦN 1: GIỌNG NÓI MẶC ĐỊNH
    # ---------------------------------------------------------
    print("\n--- 1. Danh sách giọng nói có sẵn ---")
    available_voices = tts.list_preset_voices()
    print("📋 Voices:", available_voices)
    
    # Chọn một giọng mặc định
    current_voice = tts.get_preset_voice("Binh")
    print("✅ Selected voice: Binh")


    # ---------------------------------------------------------
    # PHẦN 2: TỰ CLONE GIỌNG NÓI MỚI
    # ---------------------------------------------------------
    print("\n--- 2. Tạo giọng nói tùy chỉnh ---")
    
    # Thay bằng file .wav của bạn và nội dung tương ứng
    sample_audio = Path(__file__).parent / "example.wav"
    sample_text = "ví dụ 2. tính trung bình của dãy số."

    if sample_audio.exists():
        voice_name = "MyCustomVoice"
        print(f"🎙️ Đang clone giọng từ: {sample_audio.name}")
        
        # 'clone_voice' hỗ trợ lưu trực tiếp với tham số 'name'
        custom_voice = tts.clone_voice(
            audio_path=sample_audio,
            text=sample_text,
            name=voice_name  # <-- Tự động lưu vào hệ thống
        )
        print(f"✅ Đã tạo và lưu giọng: '{voice_name}'")
        
        current_voice = custom_voice


    # ---------------------------------------------------------
    # PHẦN 3: TỔNG HỢP GIỌNG NÓI VỚI THAM SỐ NÂNG CAO
    # ---------------------------------------------------------
    print("\n--- 3. Tổng hợp tiếng nói ---")
    text_input = "Xin chào, tôi là VieNeu-TTS. Tôi có thể giúp bạn đọc sách, hoặc clone giọng nói của bạn."
    
    print("🎧 Đang tạo âm thanh...")
    audio = tts.infer(
        text=text_input,
        voice=current_voice,
        temperature=1.0,  # 0.1 -> Ổn định, 1.0+ -> Biểu cảm
        top_k=50
    )
    sf.write("output.wav", audio, 24000)
    print("💾 Đã lưu: output.wav")

    tts.close()
    print("\n✅ Xong!")

if __name__ == "__main__":
    main()
```
*Xem thêm các script mẫu tại [main.py](main.py) ở thư mục gốc.*

---

## 🎯 3. Custom Model (LoRA, GGUF, Finetune) <a name="custom-model"></a>

VieNeu-TTS cho phép bạn tải các mô hình tùy chỉnh trực tiếp từ HuggingFace Repo hoặc đường dẫn cục bộ ngay trên giao diện Web.

- **LoRA Support:** Tự động merge LoRA vào model gốc và tăng tốc bằng **LMDeploy**.
- **GGUF Support:** Chạy mượt mà trên CPU với backend llama.cpp.
- **Private Repo:** Hỗ trợ nhập HF Token để tải các model riêng tư.

👉 Xem hướng dẫn chi tiết tại: **[docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)**

---

## 🛠️ 4. Hướng dẫn Fine-tuning <a name="finetuning"></a>

Bạn có thể tự huấn luyện VieNeu-TTS với giọng nói của chính mình hoặc dữ liệu tùy chỉnh.

- **Dễ dàng:** Sử dụng script `train.py` với cấu hình LoRA tối ưu.
- **Tài liệu:** Xem hướng dẫn từng bước tại **[finetune/README.md](finetune/README.md)**.
- **Notebook:** Trải nghiệm trực tiếp trên Google Colab với `finetune/finetune_VieNeu-TTS.ipynb`.

---

## 🔬 5. Tổng quan mô hình (Backbones) <a name="backbones"></a>

| Model Variant | Format | Thiết bị KHUYÊN DÙNG | Đặc điểm |
| :--- | :--- | :--- | :--- |
| **VieNeu-TTS** | PyTorch | NVIDIA GPU (LMDeploy) | Chất lượng tốt nhất (High Quality) |
| **VieNeu-TTS-0.3B** | PyTorch | GPU / CPU | Tốc độ cực nhanh (2x), độ trễ thấp (**Train từ đầu - Scratch**) |
| **0.3B-q8-gguf** | GGUF | CPU | Cân bằng giữa chất lượng và tốc độ |
| **0.3B-q4-gguf** | GGUF | CPU (Máy yếu) | Tốc độ xử lý nhanh nhất (Extreme Speed) |

---

## 🐋 6. Triển khai với Docker <a name="docker"></a>

Sử dụng Docker để triển khai nhanh chóng mà không cần cài đặt môi trường phức tạp.

```bash
# Chạy với CPU
docker compose --profile cpu up

# Chạy với GPU (Yêu cầu NVIDIA Container Toolkit)
docker compose --profile gpu up
```
Xem thêm chi tiết tại [docs/Deploy.md](docs/Deploy.md).

---

## 🤝 7. Hỗ trợ & Liên hệ <a name="hỗ-trợ"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Pham Nguyen Ngoc Bao](https://www.facebook.com/bao.phamnguyenngoc.5)
- **Giấy phép:** 
  - **VieNeu-TTS (0.5B):** Apache 2.0 (Sử dụng tự do).
  - **VieNeu-TTS-0.3B:** CC BY-NC 4.0 (Phi thương mại).
    - ✅ **Miễn phí:** Dành cho học sinh, sinh viên, nhà nghiên cứu hoặc các mục đích phi lợi nhuận.
    - ⚠️ **Thương mại/Doanh nghiệp:** Cần liên hệ tác giả để cấp phép (License) theo năm (Dự kiến: **5,000 USD/năm** - có thể thương lượng).

---

## 🙏 Lời cảm ơn (Acknowledgements)

Dự án này được xây dựng dựa trên các kiến trúc [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) và [NeuCodec](https://huggingface.co/neuphonic/neucodec). Cụ thể, mô hình **VieNeu-TTS (0.5B)** được fine-tune từ NeuTTS Air, trong khi mô hình **VieNeu-TTS-0.3B** là kiến trúc tùy chỉnh được huấn luyện từ đầu (trained from scratch) bằng bộ dữ liệu [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h).

---

**Made with ❤️ for the Vietnamese TTS community**
