import time
import argparse
import torch
from torch.cuda.amp import autocast

from model import TrafficClassifier


CLASSES = ["Benign", "DDoS", "PortScan", "BruteForce", "WebAttack"]


class InferenceEngine:
    def __init__(self, model_path: str, input_dim: int = 78, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = TrafficClassifier(input_dim=input_dim)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        # Warmup
        dummy = torch.zeros(1, input_dim, device=self.device)
        with torch.no_grad(), autocast():
            self.model(dummy)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> list[str]:
        x = x.to(self.device, non_blocking=True)
        with autocast():
            logits = self.model(x)
        indices = logits.argmax(dim=1).cpu().tolist()
        return [CLASSES[i] for i in indices]

    def benchmark(self, batch_size: int = 1024, n_batches: int = 100, input_dim: int = 78):
        # Use pinned memory for realistic NIC→GPU simulation
        buf = torch.zeros(batch_size, input_dim, pin_memory=True)
        latencies = []

        for _ in range(n_batches):
            x = buf.to(self.device, non_blocking=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast():
                _ = self.model(x)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1e6 / batch_size)  # µs/packet

        avg = sum(latencies) / len(latencies)
        p99 = sorted(latencies)[int(0.99 * len(latencies))]
        print(f"Inference benchmark ({batch_size} packets/batch, {n_batches} batches)")
        print(f"  Avg latency : {avg:.2f} µs/packet")
        print(f"  P99 latency : {p99:.2f} µs/packet")
        print(f"  Throughput  : {1e6/avg * 78 * 4 / 1e9:.2f} Gbps (estimated)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model.pt")
    parser.add_argument("--mode", choices=["benchmark", "predict"], default="benchmark")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    engine = InferenceEngine(args.model)
    if args.mode == "benchmark":
        engine.benchmark(batch_size=args.batch_size)
