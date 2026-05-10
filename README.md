# GPU-Accelerated-Network-Traffic-Classifier


A high-performance deep learning pipeline for real-time network intrusion detection, built with **PyTorch** and **custom CUDA kernels**. Classifies network traffic into 5 categories (Benign, DDoS, Port Scan, Brute Force, Web Attack) at line-rate speeds, achieving a **12x inference speedup** over CPU baselines.

---

## Architecture

```
Network Interface Card (NIC)
          │
          ▼  (DMA via pinned memory — zero copy)
    GPU Memory (VRAM)
          │
          ▼
  CUDA Preprocessing Kernels
  (normalization, feature extraction)
          │
          ▼
   Deep Neural Network (PyTorch)
   ┌──────────────────────────┐
   │  FC(78 → 256) + ReLU    │
   │  BatchNorm + Dropout(0.3)│
   │  FC(256 → 128) + ReLU   │
   │  BatchNorm + Dropout(0.3)│
   │  FC(128 → 64)  + ReLU   │
   │  FC(64  → 5)             │
   └──────────────────────────┘
          │
          ▼
  Classification Result
  (Benign / DDoS / PortScan / BruteForce / WebAttack)
```

---

## Results

| Metric | CPU Baseline | GPU (Ours) |
|--------|-------------|------------|
| Inference latency (per packet) | ~180 µs | **< 15 µs** |
| Throughput | ~0.8 Gbps | **10 Gbps** |
| Accuracy | 98.2% | **98.2%** |
| Speedup | 1x | **12x** |

---

## Tech Stack

- **PyTorch** — model definition, training, inference
- **CUDA / cuDNN** — custom kernels for packet preprocessing
- **NumPy / Pandas** — dataset preprocessing
- **Scikit-learn** — evaluation metrics, train/test split
- **CUDA Pinned Memory** — zero-copy DMA from NIC to GPU

---

## Project Structure

```
GPU-Accelerated-Network-Traffic-Classifier/
├── src/
│   ├── model.py            # DNN architecture
│   ├── train.py            # Training loop
│   ├── inference.py        # Real-time inference engine
│   ├── dataset.py          # Dataset loader & preprocessing
│   └── cuda_kernels.cu     # Custom CUDA preprocessing kernels
├── notebooks/
│   └── analysis.ipynb      # EDA and result visualizations
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA 11.8+
- PyTorch 2.0+ with CUDA support
- GPU with 4GB+ VRAM

```bash
git clone https://github.com/Unnati3007/GPU-Accelerated-Network-Traffic-Classifier
cd GPU-Accelerated-Network-Traffic-Classifier
pip install -r requirements.txt
```

### Dataset
Uses the **CICIDS2017** dataset (Canadian Institute for Cybersecurity).
Download from: https://www.unb.ca/cic/datasets/ids-2017.html  
Place CSVs in `data/raw/`.

### Train
```bash
python src/train.py --epochs 50 --batch-size 4096 --lr 0.001
```

### Run Inference
```bash
python src/inference.py --model checkpoints/best_model.pt --mode benchmark
```

---

## Key Implementation Details

**Pinned Memory & Zero-Copy Transfer**
```python
# Allocate pinned (page-locked) memory for fast CPU→GPU transfer
buffer = torch.zeros(BATCH_SIZE, NUM_FEATURES, pin_memory=True)
# Non-blocking async transfer — CPU doesn't wait
x = buffer.cuda(non_blocking=True)
```

**Custom CUDA Kernel for Feature Normalization**
```cuda
__global__ void normalize_features(float* data, float* mean, float* std,
                                    int n_samples, int n_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples * n_features) {
        int feat = idx % n_features;
        data[idx] = (data[idx] - mean[feat]) / (std[feat] + 1e-8f);
    }
}
```

**Mixed Precision Inference**
```python
with torch.cuda.amp.autocast():
    with torch.no_grad():
        outputs = model(x)
        predictions = torch.argmax(outputs, dim=1)
```
