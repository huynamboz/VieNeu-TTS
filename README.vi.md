# 🦜 VieNeu-TTS 

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/pnnbao97/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.5B-yellow)](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B-orange)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-0.3B--GGUF-green)](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B-q8-gguf)
[![Discord](https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white)](https://discord.gg/yJt8kzjzWZ)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V1DjG-KdmurCAhvXrxxTLsa9tteDxSVO?usp=sharing) 

<img width="899" height="615" alt="VieNeu-TTS UI" src="https://github.com/user-attachments/assets/7eb9b816-6ab7-4049-866f-f85e36cb9c6f" />

**VieNeu-TTS** là mô hình chuyển đổi văn bản thành giọng nói (TTS) tiếng Việt tiên tiến chạy trên thiết bị, hỗ trợ **instant voice cloning (clone giọng nói tức thì)**.

> [!TIP]
> **Voice Cloning:** Tất cả các biến thể mô hình (bao gồm cả GGUF) đều hỗ trợ clone giọng nói tức thì chỉ với **3-5 giây** âm thanh mẫu.

Dự án này bao gồm hai kiến trúc cốt lõi được huấn luyện trên tập dữ liệu [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h):
- **VieNeu-TTS (0.5B):** Phiên bản nâng cao được fine-tune từ kiến trúc NeuTTS Air để đạt được sự ổn định tối đa.
- **VieNeu-TTS-0.3B:** Mô hình chuyên dụng được **huấn luyện từ đầu (trained from scratch)** bằng tập dữ liệu VieNeu-TTS-1000h, mang lại tốc độ inference nhanh gấp 2 lần và độ trễ cực thấp.

Đây là một sự nâng cấp đáng kể so với VieNeu-TTS-140h trước đó với những cải tiến sau:
- **Phát âm nâng cao**: Phát âm tiếng Việt chính xác và ổn định hơn.
- **Hỗ trợ Code-switching**: Chuyển đổi mượt mà giữa tiếng Việt và tiếng Anh.
- **Clone giọng nói tốt hơn**: Độ trung thực và tính nhất quán của người nói cao hơn.
- **Tổng hợp thời gian thực**: Tạo dạng sóng 24 kHz trên CPU hoặc GPU.
- **Nhiều định dạng mô hình**: Hỗ trợ PyTorch, GGUF Q4/Q8 (tối ưu hóa cho CPU) và ONNX codec.

VieNeu-TTS cung cấp khả năng tổng hợp giọng nói sẵn sàng cho môi trường production và hoàn toàn offline.  

**Tác giả:** Phạm Nguyễn Ngọc Bao

---

[<img width="600" height="595" alt="VieNeu-TTS Demo" src="https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15" />](https://github.com/user-attachments/assets/6b32df9d-7e2e-474f-94c8-43d6fa586d15)

---

## 📌 Mục lục

1. [🦜 Cài đặt & Giao diện Web](#installation)
2. [📦 Sử dụng Python SDK](#sdk)
3. [🐳 Docker & Remote Server](#docker-remote)
4. [🎯 Mô hình tùy chỉnh](#custom-models)
5. [🛠️ Hướng dẫn Fine-tuning](#finetuning)
6. [🔬 Tổng quan mô hình](#backbones)
7. [🐋 Triển khai với Docker (Compose)](#docker)
8. [🤝 Hỗ trợ & Liên hệ](#support)

---

## 🦜 1. Cài đặt & Giao diện Web <a name="installation"></a>

Cách nhanh nhất để trải nghiệm VieNeu-TTS là thông qua giao diện Web (Gradio).

### Yêu cầu hệ thống
- **Python:** 3.12
- **eSpeak NG:** Cần thiết để xử lý âm vị.
  - **Windows:** Tải file `.msi` từ [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases).
  - **macOS:** `brew install espeak`
  - **Ubuntu/Debian:** `sudo apt install espeak-ng`
- **NVIDIA GPU (Tùy chọn):** Để đạt tốc độ tối đa qua LMDeploy hoặc GGUF tăng tốc GPU.
  - Yêu cầu **NVIDIA Driver >= 570.65** (CUDA 12.8+) hoặc cao hơn.
  - Đối với **LMDeploy**, khuyến nghị cài đặt [NVIDIA GPU Computing Toolkit](https://developer.nvidia.com/cuda-downloads).

### Các bước cài đặt
1. **Clone Repo:**
   ```bash
   git clone https://github.com/pnnbao97/VieNeu-TTS.git
   cd VieNeu-TTS
   ```

2. **Thiết lập môi trường với `uv` (Khuyến nghị):**
  - **Bước A: Cài đặt uv (nếu bạn chưa có)**
    ```bash
    # Windows:
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # Linux/macOS:
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

  - **Bước B: Cài đặt các phụ thuộc**
  
  **Lựa chọn 1: Hỗ trợ GPU (Mặc định)**
    ```bash
    uv sync
    ```
    *(Tùy chọn: Xem [Tăng tốc GGUF GPU](#gguf-gpu) nếu bạn muốn chạy mô hình GGUF trên GPU)*

  **Lựa chọn 2: Chỉ CPU (Nhẹ, không CUDA)**
    ```bash
    # Linux/macOS:
    cp pyproject.toml pyproject.toml.gpu
    cp pyproject.toml.cpu pyproject.toml
    uv sync
    
    # Windows (PowerShell/CMD):
    copy pyproject.toml pyproject.toml.gpu
    copy pyproject.toml.cpu pyproject.toml
    uv sync
    ```
    *Điều này sẽ cài đặt phiên bản PyTorch CPU (nhẹ hơn ~500MB). Để chuyển lại GPU:*
    ```bash
    # Linux/macOS: cp pyproject.toml.gpu pyproject.toml && uv sync
    # Windows: copy pyproject.toml.gpu pyproject.toml && uv sync
    ```

3. **Khởi chạy Giao diện Web:**
   ```bash
   uv run gradio_app.py
   ```
   Truy cập giao diện tại `http://127.0.0.1:7860`.

### 🚀 Tăng tốc GGUF GPU (Tùy chọn) <a name="gguf-gpu"></a>
Nếu bạn muốn sử dụng mô hình GGUF với tăng tốc GPU (llama-cpp-python), hãy làm theo các bước sau:

#### **Người dùng Windows**
Chạy lệnh sau sau khi `uv sync`:
```bash
uv pip install "https://github.com/pnnbao97/VieNeu-TTS/releases/download/llama-cpp-python-cu124/llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl"
```
*Lưu ý: Yêu cầu phiên bản NVIDIA Driver **551.61** (CUDA 12.4) hoặc mới hơn.*

#### **Người dùng Linux / macOS**
Vui lòng tham khảo [tài liệu chính thức của llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/latest/) để biết hướng dẫn cài đặt cụ thể cho phần cứng của bạn (CUDA, Metal, ROCm).

---

## 📦 2. Sử dụng Python SDK (vieneu) <a name="sdk"></a>

Tích hợp VieNeu-TTS vào các dự án phần mềm của riêng bạn.

### Cài đặt nhanh
```bash
# Windows (Tránh lỗi build llama-cpp)
pip install vieneu --extra-index-url https://pnnbao97.github.io/llama-cpp-python-v0.3.16/cpu/

# Linux / MacOS
pip install vieneu
```

### Bắt đầu nhanh (main.py)
```python
from vieneu import Vieneu

# 1. Khởi tạo (Mặc định: Tối ưu hóa cho CPU cục bộ)
tts = Vieneu() 

# Hoặc sử dụng Chế độ Remote để đạt tốc độ tối đa (xem phần Docker & Remote Server bên dưới):
# tts = Vieneu(mode="remote", api_base="http://your-server-ip:23333/v1", model_name="pnnbao-ump/VieNeu-TTS")

# 2. Tổng hợp
text = "Xin chào, tôi là VieNeu. Tôi có thể giúp bạn đọc sách, làm chatbot thời gian thực, hoặc thậm chí clone giọng nói của bạn."
audio = tts.infer(text=text)

# 3. Lưu
tts.save(audio, "output.wav")
```
*Để biết hướng dẫn đầy đủ về cloning và giọng nói tùy chỉnh, hãy xem [main.py](main.py) và [main_remote.py](main_remote.py).*

---

## 🐳 3. Docker & Remote Server <a name="docker-remote"></a>

Triển khai VieNeu-TTS dưới dạng API Server hiệu suất cao (được hỗ trợ bởi LMDeploy) chỉ bằng một lệnh duy nhất.

### 1. Chạy với Docker (Khuyến nghị)

**Yêu cầu**: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) là cần thiết để hỗ trợ GPU.

**Khởi chạy Server với Public Tunnel (Không cần mở port):**
```bash
docker run --gpus all -p 23333:23333 pnnbao/vieneu-tts:serve --tunnel
```

*   **Mặc định**: Server tải mô hình `VieNeu-TTS` để có chất lượng tối đa.
*   **Tunneling**: Image Docker bao gồm một tunnel `bore` tích hợp sẵn. Kiểm tra log container để tìm địa chỉ public của bạn (ví dụ: `bore.pub:31631`).

### 2. Sử dụng SDK (Chế độ Remote)

Sau khi server đang chạy, bạn có thể kết nối từ bất cứ đâu (Colab, Web App, v.v.) mà không cần tải mô hình nặng cục bộ:

```python
from vieneu import Vieneu

# Kết nối tới server
tts = Vieneu(
    mode='remote', 
    api_base='http://your-server-ip:23333/v1', # Hoặc URL bore tunnel
    model_name="pnnbao-ump/VieNeu-TTS"
)

# Inference cực nhanh (độ trễ thấp)
audio = tts.infer(text="Xin chào, tôi đang chạy trên một server Docker từ xa.")
tts.save(audio, "output.wav")
```

### 3. Cấu hình nâng cao

Tùy chỉnh server để chạy các phiên bản cụ thể hoặc các mô hình đã được fine-tune của riêng bạn.

**Chạy mô hình 0.3B (Nhanh hơn):**
```bash
docker run --gpus all pnnbao/vieneu-tts:serve --model pnnbao-ump/VieNeu-TTS-0.3B --tunnel
```

**Phục vụ Mô hình Fine-tuned cục bộ:**
Nếu bạn đã merge LoRA adapter, hãy mount thư mục đầu ra của bạn vào container:
```bash
# Linux / macOS
docker run --gpus all \
  -v $(pwd)/finetune/output:/workspace/models \
  pnnbao/vieneu-tts:serve \
  --model /workspace/models/merged_model --tunnel
```

*Để biết chi tiết triển khai đầy đủ, hãy xem: [main_remote.py](main_remote.py)*

---

## 🎯 4. Mô hình tùy chỉnh (LoRA, GGUF, Finetune) <a name="custom-models"></a>

VieNeu-TTS cho phép bạn tải các mô hình tùy chỉnh trực tiếp từ HuggingFace hoặc đường dẫn cục bộ thông qua giao diện Web.

- **Hỗ trợ LoRA:** Tự động merge LoRA vào mô hình gốc và tăng tốc bằng **LMDeploy**.
- **Hỗ trợ GGUF:** Chạy mượt mà trên CPU bằng backend llama.cpp.
- **Repo riêng tư:** Hỗ trợ nhập HF Token để truy cập các mô hình riêng tư.

👉 Xem hướng dẫn chi tiết tại: **[docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)**

---

## 🛠️ 5. Hướng dẫn Fine-tuning <a name="finetuning"></a>

Huấn luyện VieNeu-TTS trên giọng nói của chính bạn hoặc các tập dữ liệu tùy chỉnh.

- **Quy trình đơn giản:** Sử dụng script `train.py` với các cấu hình LoRA đã được tối ưu hóa.
- **Tài liệu:** Theo dõi hướng dẫn từng bước trong **[finetune/README.md](finetune/README.md)**.
- **Notebook:** Trải nghiệm trực tiếp trên Google Colab qua `finetune/finetune_VieNeu-TTS.ipynb`.

---

## 🔬 6. Tổng quan mô hình (Backbones) <a name="backbones"></a>

| Mô hình                 | Định dạng | Thiết bị | Chất lượng | Tốc độ                  |
| ----------------------- | --------- | -------- | ---------- | ----------------------- |
| VieNeu-TTS              | PyTorch   | GPU/CPU  | ⭐⭐⭐⭐⭐ | Rất nhanh với lmdeploy |
| VieNeu-TTS-0.3B         | PyTorch   | GPU/CPU  | ⭐⭐⭐⭐   | **Siêu nhanh (2x)**     |
| VieNeu-TTS-q8-gguf      | GGUF Q8   | CPU/GPU  | ⭐⭐⭐⭐   | Nhanh                   |
| VieNeu-TTS-q4-gguf      | GGUF Q4   | CPU/GPU  | ⭐⭐⭐     | Rất nhanh               |
| VieNeu-TTS-0.3B-q8-gguf | GGUF Q8   | CPU/GPU  | ⭐⭐⭐⭐   | **Siêu nhanh (1.5x)**   |
| VieNeu-TTS-0.3B-q4-gguf | GGUF Q4   | CPU/GPU  | ⭐⭐⭐     | **Tốc độ cực hạn (2x)** |

### 🔬 Chi tiết mô hình

- **Dữ liệu huấn luyện:** [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h) — 443,641 mẫu tiếng Việt được tinh lọc (Sử dụng cho tất cả các phiên bản).
- **Audio Codec:** NeuCodec (Bản thực thi Torch; hỗ trợ các biến thể ONNX & quantized).
- **Cửa sổ ngữ cảnh:** 2,048 token được chia sẻ bởi văn bản gợi ý và speech token.
- **Output Watermark:** Được bật theo mặc định.

---

## 🐋 7. Triển khai với Docker (Compose) <a name="docker"></a>

Triển khai nhanh chóng mà không cần thiết lập môi trường thủ công.

> **Lưu ý:** Triển khai Docker hiện chỉ hỗ trợ **GPU**. Để sử dụng CPU, vui lòng cài từ source (xem [Cài đặt & Giao diện Web](#installation)).

```bash
# Chạy với GPU (Yêu cầu NVIDIA Container Toolkit)
docker compose --profile gpu up
```
Kiểm tra [docs/Deploy.md](docs/Deploy.md) để biết thêm chi tiết.

---

## 📚 Tài liệu tham khảo

- **Dataset:** [VieNeu-TTS-1000h (Hugging Face)](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h)
- **Model 0.5B:** [pnnbao-ump/VieNeu-TTS](https://huggingface.co/pnnbao-ump/VieNeu-TTS)
- **Model 0.3B:** [pnnbao-ump/VieNeu-TTS-0.3B](https://huggingface.co/pnnbao-ump/VieNeu-TTS-0.3B)
- **Hướng dẫn LoRA:** [docs/CUSTOM_MODEL_USAGE.md](docs/CUSTOM_MODEL_USAGE.md)

---

## 🤝 8. Hỗ trợ & Liên hệ <a name="support"></a>

- **Hugging Face:** [pnnbao-ump](https://huggingface.co/pnnbao-ump)
- **Discord:** [Tham gia cộng đồng](https://discord.gg/yJt8kzjzWZ)
- **Facebook:** [Phạm Nguyễn Ngọc Bảo](https://www.facebook.com/bao.phamnguyenngoc.5)
- **Giấy phép:** 
  - **VieNeu-TTS (0.5B):** Apache 2.0 (Sử dụng tự do).
  - **VieNeu-TTS-0.3B:** CC BY-NC 4.0 (Phi thương mại).
    - ✅ **Miễn phí:** Cho sinh viên, nhà nghiên cứu và mục đích phi lợi nhuận.
    - ⚠️ **Thương mại/Doanh nghiệp:** Liên hệ tác giả để được cấp phép (Ước tính: **5,000 USD/năm** - có thể thương lượng).

---

## 📑 Trích dẫn

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

## 🙏 Lời cảm ơn

Dự án này được xây dựng dựa trên kiến trúc [NeuTTS Air](https://huggingface.co/neuphonic/neutts-air) và [NeuCodec](https://huggingface.co/neuphonic/neucodec). Cụ thể, mô hình **VieNeu-TTS (0.5B)** được fine-tune từ NeuTTS Air, trong khi mô hình **VieNeu-TTS-0.3B** là một kiến trúc tùy chỉnh được huấn luyện từ đầu sử dụng tập dữ liệu [VieNeu-TTS-1000h](https://huggingface.co/datasets/pnnbao-ump/VieNeu-TTS-1000h).

---

**Được thực hiện với ❤️ dành cho cộng đồng TTS Việt Nam**
