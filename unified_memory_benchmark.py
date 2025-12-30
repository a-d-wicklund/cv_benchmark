#!/usr/bin/env python3
"""
Unified Memory Architecture Benchmark

Demonstrates M-series advantage through iterative CPU-GPU pipeline.
This workload involves frequent CPU↔GPU data movement, where:

- Discrete GPU (x86): Requires explicit PCIe transfers (~1-2ms per large transfer)
- Unified Memory (M-series): Zero-copy, just pointer operations (~0.001ms)

The task: Iterative image processing with CPU feedback loop
- Process batch of images on GPU (blur, filters, etc.)
- Transfer to CPU for analysis (compute statistics, make decisions)
- Modify parameters based on CPU analysis
- Transfer back to GPU for next iteration
- Repeat 50-100 times

On x86 discrete GPU: PCIe transfer overhead dominates
On M4 unified memory: Essentially zero transfer cost
"""

import torch
import time
import numpy as np
import argparse
from typing import List, Tuple


class UnifiedMemoryBenchmark:
    def __init__(self, device: str = "auto", image_size: int = 2048, batch_size: int = 8):
        """
        Args:
            device: "cuda", "mps", or "auto" (auto-detect)
            image_size: Size of square images (default: 2048x2048)
            batch_size: Number of images to process (default: 8)
        """
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.device_name = "CUDA (discrete GPU)"
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.device_name = "MPS (unified memory)"
            else:
                self.device = torch.device("cpu")
                self.device_name = "CPU only"
        else:
            self.device = torch.device(device)
            self.device_name = device

        self.image_size = image_size
        self.batch_size = batch_size

        # Calculate data size
        self.data_size_mb = (batch_size * 3 * image_size * image_size * 4) / (1024**2)

        print(f"Unified Memory Architecture Benchmark")
        print(f"=" * 70)
        print(f"Device: {self.device_name}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Batch size: {batch_size}")
        print(f"Total data per iteration: {self.data_size_mb:.1f} MB")
        print(f"=" * 70)

    def create_test_data(self) -> torch.Tensor:
        """Create test image batch on GPU."""
        # Create random images (batch, channels, height, width)
        data = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        return data.to(self.device)

    def gpu_process_blur(self, images: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """Simple GPU processing: box blur using convolution."""
        # Create box blur kernel
        kernel = torch.ones(3, 1, kernel_size, kernel_size, device=self.device)
        kernel = kernel / (kernel_size * kernel_size)

        # Apply convolution (separable would be faster, but this is fine for benchmark)
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(
            images, kernel, padding=padding, groups=3
        )
        return blurred

    def gpu_process_sharpen(self, images: torch.Tensor) -> torch.Tensor:
        """GPU processing: unsharp mask sharpening."""
        # Simple sharpening kernel
        kernel = torch.tensor([
            [[-1, -1, -1],
             [-1,  9, -1],
             [-1, -1, -1]]
        ], dtype=torch.float32, device=self.device)
        kernel = kernel.unsqueeze(0).repeat(3, 1, 1, 1) / 9.0

        sharpened = torch.nn.functional.conv2d(
            images, kernel, padding=1, groups=3
        )
        return sharpened

    def cpu_analyze(self, images: torch.Tensor) -> Tuple[float, float]:
        """
        CPU analysis: compute statistics (requires CPU access).
        This forces a GPU→CPU transfer on discrete GPUs.
        """
        # Transfer to CPU (expensive on discrete GPU, free on unified memory)
        cpu_data = images.cpu()

        # Compute statistics
        mean = float(cpu_data.mean())
        std = float(cpu_data.std())

        return mean, std

    def cpu_modify_params(self, mean: float, std: float) -> dict:
        """
        CPU decision making: choose next operation based on statistics.
        Simulates the kind of CPU logic you'd have in a real pipeline.
        """
        # Contrived decision logic
        if abs(mean) > 0.5:
            return {"operation": "blur", "kernel_size": 7}
        elif std > 1.0:
            return {"operation": "blur", "kernel_size": 5}
        else:
            return {"operation": "sharpen"}

    def benchmark_iterative_pipeline(self, iterations: int = 100) -> dict:
        """
        Main benchmark: Iterative CPU-GPU pipeline.

        Each iteration:
        1. Process on GPU (blur or sharpen)
        2. Transfer to CPU for analysis (GPU→CPU)
        3. Compute statistics on CPU
        4. Make decision on CPU
        5. Transfer back to GPU (CPU→GPU) - implicitly when next GPU op starts
        6. Repeat

        On discrete GPU: Steps 2 and 5 require explicit PCIe transfers
        On unified memory: Steps 2 and 5 are essentially free (pointer ops)
        """
        print(f"\nRunning iterative CPU-GPU pipeline ({iterations} iterations)...")

        # Create initial data on GPU
        images = self.create_test_data()

        # Warmup
        for _ in range(5):
            images = self.gpu_process_blur(images)
            mean, std = self.cpu_analyze(images)

        # Synchronize before timing
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        # Timed run
        times = {
            "total": 0.0,
            "gpu_processing": 0.0,
            "cpu_analysis": 0.0,
            "iterations": iterations
        }

        start_total = time.perf_counter()

        for i in range(iterations):
            # GPU processing
            gpu_start = time.perf_counter()

            # Get parameters from previous iteration (or use defaults)
            if i == 0:
                params = {"operation": "blur", "kernel_size": 5}

            # Apply operation on GPU
            if params["operation"] == "blur":
                images = self.gpu_process_blur(images, params.get("kernel_size", 5))
            else:
                images = self.gpu_process_sharpen(images)

            # Sync GPU
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()

            gpu_time = time.perf_counter() - gpu_start
            times["gpu_processing"] += gpu_time

            # CPU analysis (forces GPU→CPU transfer on discrete GPU)
            cpu_start = time.perf_counter()
            mean, std = self.cpu_analyze(images)
            params = self.cpu_modify_params(mean, std)
            cpu_time = time.perf_counter() - cpu_start
            times["cpu_analysis"] += cpu_time

            # Move back to GPU (happens implicitly at next GPU operation)
            # On discrete GPU, this is when the CPU→GPU transfer happens
            images = images.to(self.device)

        end_total = time.perf_counter()
        times["total"] = end_total - start_total

        # Calculate transfer overhead
        times["transfer_overhead"] = times["total"] - times["gpu_processing"] - times["cpu_analysis"]

        return times

    def benchmark_pure_transfer(self, iterations: int = 100) -> dict:
        """
        Pure transfer benchmark: Just measure GPU↔CPU transfer cost.
        This isolates the memory architecture difference.
        """
        print(f"\nRunning pure transfer benchmark ({iterations} iterations)...")

        # Create data on GPU
        gpu_data = self.create_test_data()

        # Warmup
        for _ in range(5):
            cpu_data = gpu_data.cpu()
            gpu_data = cpu_data.to(self.device)

        # Synchronize
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.synchronize()

        # Time GPU→CPU transfers
        start = time.perf_counter()
        for _ in range(iterations):
            cpu_data = gpu_data.cpu()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
        gpu_to_cpu_time = time.perf_counter() - start

        # Time CPU→GPU transfers
        start = time.perf_counter()
        for _ in range(iterations):
            gpu_data = cpu_data.to(self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elif self.device.type == "mps":
                torch.mps.synchronize()
        cpu_to_gpu_time = time.perf_counter() - start

        return {
            "gpu_to_cpu_total": gpu_to_cpu_time,
            "cpu_to_gpu_total": cpu_to_gpu_time,
            "gpu_to_cpu_per_iter": (gpu_to_cpu_time / iterations) * 1000,  # ms
            "cpu_to_gpu_per_iter": (cpu_to_gpu_time / iterations) * 1000,  # ms
            "iterations": iterations
        }

    def print_results(self, pipeline_times: dict, transfer_times: dict):
        """Print formatted results."""
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        print(f"\nIterative Pipeline ({pipeline_times['iterations']} iterations):")
        print(f"  Total time:          {pipeline_times['total']*1000:8.2f} ms ({pipeline_times['total']/pipeline_times['iterations']*1000:.3f} ms/iter)")
        print(f"  GPU processing:      {pipeline_times['gpu_processing']*1000:8.2f} ms ({pipeline_times['gpu_processing']/pipeline_times['iterations']*1000:.3f} ms/iter)")
        print(f"  CPU analysis:        {pipeline_times['cpu_analysis']*1000:8.2f} ms ({pipeline_times['cpu_analysis']/pipeline_times['iterations']*1000:.3f} ms/iter)")
        print(f"  Transfer overhead:   {pipeline_times['transfer_overhead']*1000:8.2f} ms ({pipeline_times['transfer_overhead']/pipeline_times['iterations']*1000:.3f} ms/iter)")
        print(f"  Transfer % of total: {pipeline_times['transfer_overhead']/pipeline_times['total']*100:8.1f}%")

        print(f"\nPure Transfer Benchmark ({transfer_times['iterations']} iterations):")
        print(f"  GPU→CPU per transfer: {transfer_times['gpu_to_cpu_per_iter']:8.3f} ms  ({self.data_size_mb/transfer_times['gpu_to_cpu_per_iter']*1000:.1f} GB/s)")
        print(f"  CPU→GPU per transfer: {transfer_times['cpu_to_gpu_per_iter']:8.3f} ms  ({self.data_size_mb/transfer_times['cpu_to_gpu_per_iter']*1000:.1f} GB/s)")
        print(f"  Round-trip transfer:  {transfer_times['gpu_to_cpu_per_iter'] + transfer_times['cpu_to_gpu_per_iter']:8.3f} ms")

        print("\n" + "=" * 70)
        print("INTERPRETATION")
        print("=" * 70)

        if "unified" in self.device_name.lower() or "mps" in self.device_name.lower():
            print("\n✓ Unified Memory Architecture (M-series)")
            print("  - Transfer overhead should be minimal (<1% of total time)")
            print("  - GPU↔CPU 'transfers' are just pointer operations")
            print("  - Expected: <0.1ms per transfer")
        else:
            print("\n⚠ Discrete GPU Architecture")
            print("  - Transfer overhead can be significant (20-50% of total time)")
            print("  - Requires explicit PCIe transfers")
            print("  - Expected: 1-5ms per transfer for this data size")


def main():
    parser = argparse.ArgumentParser(description="Unified Memory Architecture Benchmark")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: 'cuda', 'mps', or 'auto' (default: auto)")
    parser.add_argument("--image-size", type=int, default=2048,
                       help="Image size (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size (default: 8)")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of iterations (default: 100)")

    args = parser.parse_args()

    # Create benchmark
    benchmark = UnifiedMemoryBenchmark(
        device=args.device,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    # Run benchmarks
    pipeline_times = benchmark.benchmark_iterative_pipeline(args.iterations)
    transfer_times = benchmark.benchmark_pure_transfer(args.iterations)

    # Print results
    benchmark.print_results(pipeline_times, transfer_times)


if __name__ == "__main__":
    main()
