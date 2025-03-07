#!/usr/bin/env python3
"""
Ollama Benchmark Script

This script benchmarks Ollama models by comparing serial vs. parallel throughput.
It tests multiple models to analyze how model size affects parallelization benefits.
"""

import argparse
import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from typing import List, Dict, Any, Tuple
import statistics

# Configuration
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODELS = ["mistral", "llama2", "gemma:2b", "phi"]
DEFAULT_PROMPT = "Explain the concept of parallelism in computing in a paragraph."
DEFAULT_NUM_REQUESTS = 5
DEFAULT_MAX_WORKERS = 4


class OllamaBenchmark:
    def __init__(self,
                 ollama_url: str = DEFAULT_OLLAMA_URL,
                 models: List[str] = DEFAULT_MODELS,
                 prompt: str = DEFAULT_PROMPT,
                 num_requests: int = DEFAULT_NUM_REQUESTS,
                 max_workers: int = DEFAULT_MAX_WORKERS,
                 use_all_models: bool = False):
        """Initialize the benchmark with configuration parameters."""
        self.ollama_url = ollama_url
        self.prompt = prompt
        self.num_requests = num_requests
        self.max_workers = max_workers
        self.results = {}

        # If use_all_models flag is set, discover all available models
        if use_all_models:
            self.models = self.discover_all_models()
            if not self.models:
                print("No models found. Using default models instead.")
                self.models = models
        else:
            self.models = models

    def check_model_availability(self, model: str) -> bool:
        """Check if a model is available in the local Ollama instance."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            return model in models
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False

    def send_request(self, model: str) -> Tuple[float, int, str]:
        """Send a single request to Ollama and measure the time taken."""
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": self.prompt, "stream": False}
            )
            response.raise_for_status()
            result = response.json()
            elapsed = time.time() - start_time
            token_count = len(result.get("response", "").split())
            return elapsed, token_count, "success"
        except Exception as e:
            elapsed = time.time() - start_time
            return elapsed, 0, f"error: {str(e)}"

    def run_serial_benchmark(self, model: str) -> Dict[str, Any]:
        """Run serial benchmark for a specific model."""
        print(f"\nRunning serial benchmark for {model}...")
        times = []
        tokens = []
        errors = 0

        total_start = time.time()
        for i in range(self.num_requests):
            print(f"  Request {i + 1}/{self.num_requests}...", end="", flush=True)
            elapsed, token_count, status = self.send_request(model)
            times.append(elapsed)

            if "error" in status:
                errors += 1
                print(f" {status}")
            else:
                tokens.append(token_count)
                print(f" completed in {elapsed:.2f}s, {token_count} tokens")

        total_time = time.time() - total_start

        return {
            "total_time": total_time,
            "avg_request_time": statistics.mean(times) if times else 0,
            "median_request_time": statistics.median(times) if times else 0,
            "min_request_time": min(times) if times else 0,
            "max_request_time": max(times) if times else 0,
            "avg_tokens": statistics.mean(tokens) if tokens else 0,
            "total_tokens": sum(tokens),
            "requests_per_second": self.num_requests / total_time if total_time > 0 else 0,
            "tokens_per_second": sum(tokens) / total_time if total_time > 0 else 0,
            "errors": errors,
        }

    def run_parallel_benchmark(self, model: str) -> Dict[str, Any]:
        """Run parallel benchmark for a specific model."""
        print(f"\nRunning parallel benchmark for {model} with {self.max_workers} workers...")

        total_start = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.send_request, model) for _ in range(self.num_requests)]
            results = [future.result() for future in futures]

        times = [r[0] for r in results]
        tokens = [r[1] for r in results if r[1] > 0]
        errors = sum(1 for r in results if "error" in r[2])

        for i, (elapsed, token_count, status) in enumerate(results):
            print(f"  Request {i + 1}: ", end="")
            if "error" in status:
                print(f"{status}")
            else:
                print(f"completed in {elapsed:.2f}s, {token_count} tokens")

        total_time = time.time() - total_start

        return {
            "total_time": total_time,
            "avg_request_time": statistics.mean(times) if times else 0,
            "median_request_time": statistics.median(times) if times else 0,
            "min_request_time": min(times) if times else 0,
            "max_request_time": max(times) if times else 0,
            "avg_tokens": statistics.mean(tokens) if tokens else 0,
            "total_tokens": sum(tokens),
            "requests_per_second": self.num_requests / total_time if total_time > 0 else 0,
            "tokens_per_second": sum(tokens) / total_time if total_time > 0 else 0,
            "errors": errors,
        }

    def discover_all_models(self) -> List[str]:
        """Discover all available models in the local Ollama instance."""
        print("Discovering all available models...")
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]

            if models:
                print(f"Found {len(models)} models: {', '.join(models)}")
            else:
                print("No models found in Ollama instance.")

            return models
        except Exception as e:
            print(f"Error discovering models: {e}")
            return []

    def preload_model(self, model: str) -> bool:
        """Preload a model to ensure it's in memory before benchmarking."""
        print(f"Preloading model {model}...")
        try:
            # Send a simple request to load the model into memory
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": "Hello", "stream": False}
            )
            response.raise_for_status()
            print(f"Model {model} successfully preloaded.")
            return True
        except Exception as e:
            print(f"Error preloading model {model}: {e}")
            return False

    def run_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for all specified models."""
        results = {}

        print(f"Ollama Benchmark")
        print(f"===============")
        print(f"URL: {self.ollama_url}")
        print(f"Requests per model: {self.num_requests}")
        print(f"Max parallel workers: {self.max_workers}")
        print(f"Prompt: '{self.prompt}'")
        print(f"Number of models to benchmark: {len(self.models)}")

        for model in self.models:
            print(f"\n{'=' * 50}")
            print(f"Benchmarking model: {model}")
            print(f"{'=' * 50}")

            # Check if model is available (only needed when not using --allmodels)
            if not self.check_model_availability(model):
                print(f"Model {model} is not available in your local Ollama instance.")
                print(f"Try pulling it first with: ollama pull {model}")
                results[model] = {"error": "Model not available"}
                continue

            # Preload the model to avoid first-request loading time
            if not self.preload_model(model):
                print(f"Warning: Failed to preload model {model}. Benchmarks may include model loading time.")
                results[model] = {"error": "Failed to preload model"}
                continue

            model_results = {
                "serial": self.run_serial_benchmark(model),
                "parallel": self.run_parallel_benchmark(model)
            }

            # Calculate performance improvement
            serial_time = model_results["serial"]["total_time"]
            parallel_time = model_results["parallel"]["total_time"]
            speedup = serial_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / self.max_workers

            model_results["comparison"] = {
                "speedup": speedup,
                "efficiency": efficiency,
                "improvement_percentage": (1 - parallel_time / serial_time) * 100 if serial_time > 0 else 0
            }

            results[model] = model_results

            # Display summary for this model
            self.display_model_summary(model, model_results)

        self.results = results
        return results

    def display_model_summary(self, model: str, results: Dict[str, Any]) -> None:
        """Display a summary of benchmark results for a model."""
        comparison = results.get("comparison", {})
        serial = results.get("serial", {})
        parallel = results.get("parallel", {})

        print(f"\nSummary for {model}:")
        print(
            f"  Serial:   {serial['total_time']:.2f}s total, {serial['requests_per_second']:.2f} req/s, {serial['tokens_per_second']:.2f} tokens/s")
        print(
            f"  Parallel: {parallel['total_time']:.2f}s total, {parallel['requests_per_second']:.2f} req/s, {parallel['tokens_per_second']:.2f} tokens/s")
        print(f"  Speedup: {comparison.get('speedup', 0):.2f}x (Efficiency: {comparison.get('efficiency', 0):.2f})")
        print(f"  Improvement: {comparison.get('improvement_percentage', 0):.1f}%")

    def save_results(self, filename: str = "ollama_benchmark_results.json") -> None:
        """Save the benchmark results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")

    def plot_results(self) -> None:
        """Generate visual plots of the results."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if not self.results:
                print("No results to plot.")
                return

            # Create comparison plots
            models = list(self.results.keys())
            serial_times = [self.results[m].get("serial", {}).get("total_time", 0) for m in models]
            parallel_times = [self.results[m].get("parallel", {}).get("total_time", 0) for m in models]
            speedups = [self.results[m].get("comparison", {}).get("speedup", 0) for m in models]

            # Filter out models with errors
            valid_indices = [i for i, m in enumerate(models) if
                             isinstance(self.results[m], dict) and "error" not in self.results[m]]
            models = [models[i] for i in valid_indices]
            serial_times = [serial_times[i] for i in valid_indices]
            parallel_times = [parallel_times[i] for i in valid_indices]
            speedups = [speedups[i] for i in valid_indices]

            if not models:
                print("No valid benchmark results to plot.")
                return

            # Plot execution times
            plt.figure(figsize=(12, 6))
            x = np.arange(len(models))
            width = 0.35

            plt.bar(x - width / 2, serial_times, width, label='Serial')
            plt.bar(x + width / 2, parallel_times, width, label='Parallel')

            plt.xlabel('Models')
            plt.ylabel('Total Execution Time (s)')
            plt.title('Serial vs Parallel Execution Time')
            plt.xticks(x, models)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save the plot
            plt.savefig('ollama_benchmark_times.png')
            print("Execution time plot saved as ollama_benchmark_times.png")

            # Plot speedup
            plt.figure(figsize=(12, 6))
            plt.bar(models, speedups, color='green')
            plt.axhline(y=1, color='r', linestyle='-', alpha=0.5)
            plt.axhline(y=self.max_workers, color='b', linestyle='--', alpha=0.5)

            plt.xlabel('Models')
            plt.ylabel('Speedup (X times)')
            plt.title(f'Parallel Speedup (with {self.max_workers} workers)')
            plt.grid(True, linestyle='--', alpha=0.7)

            # Save the plot
            plt.savefig('ollama_benchmark_speedup.png')
            print("Speedup plot saved as ollama_benchmark_speedup.png")

        except ImportError:
            print("Matplotlib is required for plotting. Install it with: pip install matplotlib")
        except Exception as e:
            print(f"Error generating plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Ollama models for serial vs. parallel performance')
    parser.add_argument('--url', default=DEFAULT_OLLAMA_URL, help='Ollama API URL')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Models to benchmark')
    parser.add_argument('--allmodels', action='store_true', help='Discover and benchmark all available models')
    parser.add_argument('--prompt', default=DEFAULT_PROMPT, help='Prompt to use for testing')
    parser.add_argument('--requests', type=int, default=DEFAULT_NUM_REQUESTS, help='Number of requests per model')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS, help='Maximum number of parallel workers')
    parser.add_argument('--plot', action='store_true', help='Generate plots of the results')
    parser.add_argument('--output', default='ollama_benchmark_results.json', help='Output file for results')

    args = parser.parse_args()

    # If both --models and --allmodels are specified, warn the user
    if args.allmodels and args.models != DEFAULT_MODELS:
        print("Warning: Both --models and --allmodels specified. Using --allmodels.")

    benchmark = OllamaBenchmark(
        ollama_url=args.url,
        models=args.models,
        prompt=args.prompt,
        num_requests=args.requests,
        max_workers=args.workers,
        use_all_models=args.allmodels
    )

    benchmark.run_benchmarks()
    benchmark.save_results(args.output)

    if args.plot:
        benchmark.plot_results()


if __name__ == "__main__":
    main()
