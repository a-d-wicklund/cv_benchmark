#!/usr/bin/env python3
"""
Traditional Computer Vision Benchmark Suite
Benchmarks various OpenCV operations for performance comparison across systems.
"""

import cv2
import numpy as np
import time
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Callable


def benchmark_operation(func: Callable, name: str, iterations: int = 10) -> Dict:
    """Run an operation multiple times and return timing statistics."""
    times = []
    result = None

    # Warm-up run
    result = func()

    # Timed runs
    for _ in range(iterations):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        "name": name,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "iterations": iterations
    }


def download_test_image(output_path: str = "test_image.jpg") -> bool:
    """Download a test image from the internet."""
    try:
        print(f"Downloading test image to {output_path}...")
        result = subprocess.run(
            ["curl", "-L", "-o", output_path,
             "https://unsplash.com/photos/8xAA0f9yQnE/download?force=true&w=1920"],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0 and Path(output_path).exists():
            print(f"Successfully downloaded test image")
            return True
        return False
    except Exception as e:
        print(f"Failed to download image: {e}")
        return False


def create_sample_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Create a sample image with various features for testing."""
    print(f"Creating sample image ({width}x{height})...")
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some colorful shapes and patterns
    cv2.rectangle(img, (100, 100), (500, 500), (255, 0, 0), -1)
    cv2.circle(img, (800, 300), 150, (0, 255, 0), -1)
    cv2.ellipse(img, (1200, 600), (200, 100), 45, 0, 360, (0, 0, 255), -1)

    # Add some noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Add some lines
    for i in range(0, width, 50):
        cv2.line(img, (i, 0), (i, height), (128, 128, 128), 1)

    return img


class CVBenchmark:
    def __init__(self, image_path: str = None, iterations: int = 10):
        self.iterations = iterations
        self.results = []

        # Determine which image to use
        if image_path:
            # User specified an image path
            if not Path(image_path).exists():
                raise ValueError(f"Specified image not found: {image_path}")
            target_path = image_path
        else:
            # No image specified - check for test_image.jpg
            target_path = "test_image.jpg"
            if not Path(target_path).exists():
                # Download test image if it doesn't exist
                if not download_test_image(target_path):
                    # Download failed - use synthetic image
                    print("Using synthetic image as fallback")
                    self.img = create_sample_image()
                    print(f"Image shape: {self.img.shape}")
                    print(f"Running {iterations} iterations per operation...\n")
                    return

        # Load the image
        print(f"Loading image from {target_path}...")
        self.img = cv2.imread(target_path)
        if self.img is None:
            raise ValueError(f"Failed to load image from {target_path}")

        print(f"Image shape: {self.img.shape}")
        print(f"Running {iterations} iterations per operation...\n")

    def run_all_benchmarks(self):
        """Run all benchmark categories."""
        print("=" * 70)
        print("TRADITIONAL COMPUTER VISION BENCHMARK")
        print("=" * 70)

        self.benchmark_io_preprocessing()
        self.benchmark_filtering()
        self.benchmark_edge_detection()
        self.benchmark_morphological()
        self.benchmark_feature_detection()
        self.benchmark_pyramids()

        self.print_summary()
        self.save_results()

    def benchmark_io_preprocessing(self):
        """Benchmark I/O and basic preprocessing operations."""
        print("\n[1/6] Image I/O and Preprocessing")
        print("-" * 70)

        # Resize operations
        sizes = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
        for width, height in sizes:
            result = benchmark_operation(
                lambda w=width, h=height: cv2.resize(self.img, (w, h), interpolation=cv2.INTER_LINEAR),
                f"Resize to {width}x{height} (bilinear)",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Color space conversions
        conversions = [
            (cv2.COLOR_BGR2GRAY, "BGR to Grayscale"),
            (cv2.COLOR_BGR2HSV, "BGR to HSV"),
            (cv2.COLOR_BGR2LAB, "BGR to LAB"),
        ]

        for code, name in conversions:
            result = benchmark_operation(
                lambda c=code: cv2.cvtColor(self.img, c),
                name,
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

    def benchmark_filtering(self):
        """Benchmark various filtering operations."""
        print("\n[2/6] Filtering Operations")
        print("-" * 70)

        # Gaussian blur - test small and large kernels
        for ksize in [5, 21]:
            result = benchmark_operation(
                lambda k=ksize: cv2.GaussianBlur(self.img, (k, k), 0),
                f"Gaussian Blur (kernel {ksize}x{ksize})",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Bilateral filter - edge-preserving, computationally expensive
        for ksize in [5, 15]:
            result = benchmark_operation(
                lambda k=ksize: cv2.bilateralFilter(self.img, k, 75, 75),
                f"Bilateral Filter (kernel {ksize}x{ksize})",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Median blur - non-linear filter, good for salt-and-pepper noise
        for ksize in [5, 9]:
            result = benchmark_operation(
                lambda k=ksize: cv2.medianBlur(self.img, k),
                f"Median Blur (kernel {ksize}x{ksize})",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

    def benchmark_edge_detection(self):
        """Benchmark edge and corner detection algorithms."""
        print("\n[3/6] Edge and Corner Detection")
        print("-" * 70)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Canny edge detection
        result = benchmark_operation(
            lambda: cv2.Canny(gray, 50, 150),
            "Canny Edge Detection",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Sobel operator (gradient-based edge detection)
        result = benchmark_operation(
            lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5),
            "Sobel X (kernel 5x5)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Harris corner detection
        result = benchmark_operation(
            lambda: cv2.cornerHarris(gray, 2, 3, 0.04),
            "Harris Corner Detection",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Shi-Tomasi corner detection
        result = benchmark_operation(
            lambda: cv2.goodFeaturesToTrack(gray, 100, 0.01, 10),
            "Shi-Tomasi Corner Detection (100 corners)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

    def benchmark_morphological(self):
        """Benchmark morphological operations."""
        print("\n[4/6] Morphological Operations")
        print("-" * 70)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Test small and large kernels for basic operations
        for ksize in [5, 21]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

            # Erosion
            result = benchmark_operation(
                lambda k=kernel: cv2.erode(binary, k, iterations=1),
                f"Erosion (kernel {ksize}x{ksize})",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

            # Dilation
            result = benchmark_operation(
                lambda k=kernel: cv2.dilate(binary, k, iterations=1),
                f"Dilation (kernel {ksize}x{ksize})",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Compound operations (erosion + dilation)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        result = benchmark_operation(
            lambda: cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
            "Opening (kernel 9x9)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        result = benchmark_operation(
            lambda: cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
            "Closing (kernel 9x9)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

    def benchmark_feature_detection(self):
        """Benchmark feature detection and descriptor extraction."""
        print("\n[5/6] Feature Detection and Extraction")
        print("-" * 70)

        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # ORB (Oriented FAST and Rotated BRIEF)
        orb = cv2.ORB_create(nfeatures=500)
        result = benchmark_operation(
            lambda: orb.detectAndCompute(gray, None),
            "ORB (500 features)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        orb_large = cv2.ORB_create(nfeatures=2000)
        result = benchmark_operation(
            lambda: orb_large.detectAndCompute(gray, None),
            "ORB (2000 features)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # AKAZE (Accelerated-KAZE)
        akaze = cv2.AKAZE_create()
        result = benchmark_operation(
            lambda: akaze.detectAndCompute(gray, None),
            "AKAZE",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

    def benchmark_pyramids(self):
        """Benchmark image pyramid construction."""
        print("\n[6/6] Image Pyramids")
        print("-" * 70)

        # Gaussian pyramid - test shallow and deep pyramids
        for num_levels in [3, 7]:
            def build_gaussian_pyramid(n=num_levels):
                pyramid = [self.img]
                for _ in range(n):
                    pyramid.append(cv2.pyrDown(pyramid[-1]))
                return pyramid

            result = benchmark_operation(
                build_gaussian_pyramid,
                f"Gaussian Pyramid ({num_levels} levels)",
                self.iterations
            )
            self.results.append(result)
            print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

        # Laplacian pyramid
        def build_laplacian_pyramid():
            gaussian = [self.img]
            for _ in range(5):
                gaussian.append(cv2.pyrDown(gaussian[-1]))

            laplacian = []
            for i in range(len(gaussian) - 1):
                size = (gaussian[i].shape[1], gaussian[i].shape[0])
                expanded = cv2.pyrUp(gaussian[i + 1], dstsize=size)
                laplacian.append(cv2.subtract(gaussian[i], expanded))

            return laplacian

        result = benchmark_operation(
            build_laplacian_pyramid,
            "Laplacian Pyramid (5 levels)",
            self.iterations
        )
        self.results.append(result)
        print(f"  {result['name']:<50} {result['mean_ms']:>8.2f} ± {result['std_ms']:>6.2f} ms")

    def print_summary(self):
        """Print a summary of the top slowest operations."""
        print("\n" + "=" * 70)
        print("SUMMARY - Top 10 Slowest Operations")
        print("=" * 70)

        sorted_results = sorted(self.results, key=lambda x: x['mean_ms'], reverse=True)[:10]

        for i, result in enumerate(sorted_results, 1):
            print(f"{i:2d}. {result['name']:<50} {result['mean_ms']:>8.2f} ms")

    def save_results(self):
        """Save results to a JSON file."""
        output_file = "benchmark_results.json"

        import platform
        metadata = {
            "platform": platform.system(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "opencv_version": cv2.__version__,
            "image_shape": self.img.shape,
            "iterations": self.iterations
        }

        output = {
            "metadata": metadata,
            "results": self.results
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Traditional CV Benchmark Suite")
    parser.add_argument("--image", type=str, help="Path to input image (optional, will generate one if not provided)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per operation (default: 10)")

    args = parser.parse_args()

    try:
        benchmark = CVBenchmark(image_path=args.image, iterations=args.iterations)
        benchmark.run_all_benchmarks()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
