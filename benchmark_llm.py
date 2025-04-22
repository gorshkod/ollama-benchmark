#!/usr/bin/env python3
"""
LLM Benchmark Tool for Raspberry Pi 5 using Ollama

This script benchmarks LLMs on key metrics:
- Latency (response time)
- Throughput (tokens per second)
- Resource utilization (CPU, RAM, GPU if available)
- Response accuracy (via predefined test cases)
"""

import argparse
import json
import os
import psutil
import requests
import statistics
import time
import threading
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api"

class OllamaBenchmark:
    def __init__(self, model_name: str, num_threads: int = 1, 
                 num_requests: int = 10, prompt_size: str = "medium",
                 output_file: str = None, accuracy_test: bool = False):
        """
        Initialize the benchmark settings.
        
        Args:
            model_name: Name of the Ollama model to benchmark
            num_threads: Number of concurrent threads for throughput testing
            num_requests: Number of requests to send for each test
            prompt_size: Size of the prompt ("small", "medium", "large")
            output_file: File to save benchmark results
            accuracy_test: Whether to perform accuracy tests
        """
        self.model_name = model_name
        self.num_threads = num_threads
        self.num_requests = num_requests
        self.prompt_size = prompt_size
        self.output_file = output_file
        self.accuracy_test = accuracy_test
        self.results = {}
        
        # Set up test prompts based on size
        self.prompts = self._get_prompts()
        
        # Set up accuracy test questions and expected answers
        self.accuracy_tests = self._get_accuracy_tests()
        
        # Resource usage tracker
        self.resource_usage = {
            "cpu": [],
            "memory": [],
            "start_time": 0,
            "end_time": 0
        }
        
        # String buffer to capture console output
        self.console_output = []
        
    def _print(self, *args, **kwargs):
        """Custom print function to capture output while still printing to console."""
        # Convert args to string
        output = " ".join(str(arg) for arg in args)
        
        # Add kwargs if any
        if kwargs.get("end", "\n") != "\n":
            output += kwargs.get("end", "")
        
        # Store output
        self.console_output.append(output)
        
        # Print to console as usual
        print(*args, **kwargs)
        
    def _get_prompts(self) -> Dict[str, List[str]]:
        """Generate prompts of different sizes for testing."""
        prompts = {
            "small": [
                "What is 2+2?",
                "Name the capital of France.",
                "What is the color of the sky?",
                "Who wrote Hamlet?",
                "What is the chemical symbol for water?"
            ],
            "medium": [
                "Explain how photosynthesis works in a few sentences.",
                "Summarize the plot of The Great Gatsby in a paragraph.",
                "Describe the process of making bread from scratch.",
                "What are the key differences between Python and JavaScript?",
                "Explain the theory of relativity in simple terms."
            ],
            "large": [
                "Write a 500-word essay on the impact of artificial intelligence on society.",
                "Provide a detailed analysis of climate change causes and potential solutions.",
                "Describe the complete historical timeline of World War II with key events.",
                "Write a comprehensive guide to machine learning algorithms and their applications.",
                "Analyze the economic impact of the COVID-19 pandemic on global markets."
            ]
        }
        return prompts
    
    def _get_accuracy_tests(self) -> List[Dict[str, Any]]:
        """Define questions and answers for accuracy testing."""
        return [
            {
                "question": "What is the capital of France?",
                "expected_keywords": ["paris", "france", "capital", "city"]
            },
            {
                "question": "Who wrote the play Romeo and Juliet?",
                "expected_keywords": ["shakespeare", "william", "playwright", "wrote", "author"]
            },
            {
                "question": "What is the formula for water?",
                "expected_keywords": ["h2o", "hydrogen", "oxygen", "molecule", "formula", "water"]
            },
            {
                "question": "What is the square root of 144?",
                "expected_keywords": ["12", "square", "root", "144"]
            },
            {
                "question": "In what year did World War II end?",
                "expected_keywords": ["1945", "world war", "end", "concluded"]
            }
        ]
    
    def _start_resource_monitoring(self):
        """Start monitoring system resources."""
        self.resource_usage["start_time"] = time.time()
        self.stop_monitoring = False
        
        def monitor_resources():
            while not self.stop_monitoring:
                self.resource_usage["cpu"].append(psutil.cpu_percent(interval=0.5))
                self.resource_usage["memory"].append(psutil.virtual_memory().percent)
                time.sleep(0.5)
        
        self.monitor_thread = threading.Thread(target=monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _stop_resource_monitoring(self):
        """Stop monitoring system resources."""
        self.stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        self.resource_usage["end_time"] = time.time()
    
    def call_ollama(self, prompt: str) -> Tuple[Dict[str, Any], float, int]:
        """
        Make a single call to Ollama API.
        
        Returns:
            Tuple of (response_data, time_taken, token_count)
        """
        url = f"{OLLAMA_API}/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        try:
            response_data = response.json()
            token_count = response_data.get('eval_count', 0) + response_data.get('prompt_eval_count', 0)
            return response_data, end_time - start_time, token_count
        except json.JSONDecodeError:
            print(f"Error decoding response: {response.text}")
            return {"error": "Failed to decode response"}, end_time - start_time, 0
    
    def measure_latency(self) -> Dict[str, Any]:
        """
        Measure response latency for different prompt sizes.
        
        Returns:
            Dictionary with latency statistics
        """
        self._print(f"\n🕒 Measuring latency for {self.model_name}...")
        results = {"small": [], "medium": [], "large": []}
        prompt_response_data = {"small": [], "medium": [], "large": []}
        
        # Use selected prompt size or all sizes for comprehensive testing
        prompt_sizes = [self.prompt_size] if self.prompt_size in results else list(results.keys())
        
        for size in prompt_sizes:
            self._print(f"  Testing with {size} prompts...")
            prompts = self.prompts[size]
            
            for i in range(min(self.num_requests, len(prompts))):
                prompt = prompts[i % len(prompts)]
                response, latency, token_count = self.call_ollama(prompt)
                results[size].append(latency)
                
                # Store prompt, response, and token count
                prompt_response_data[size].append({
                    "prompt": prompt,
                    "response": response.get('response', ''),
                    "prompt_tokens": response.get('prompt_eval_count', 0),
                    "completion_tokens": response.get('eval_count', 0),
                    "total_tokens": token_count
                })
                
                self._print(f"    Request {i+1}/{min(self.num_requests, len(prompts))}: {latency:.2f}s, {token_count} tokens")
        
        # Calculate statistics
        stats = {}
        for size, latencies in results.items():
            if latencies:
                stats[size] = {
                    "min": min(latencies),
                    "max": max(latencies),
                    "mean": statistics.mean(latencies),
                    "median": statistics.median(latencies),
                    "p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                    "raw": latencies,
                    "prompt_response_data": prompt_response_data[size]
                }
        
        return stats
    
    def measure_throughput(self) -> Dict[str, Any]:
        """
        Measure throughput using concurrent requests.
        
        Returns:
            Dictionary with throughput statistics
        """
        self._print(f"\n⚡ Measuring throughput for {self.model_name} with {self.num_threads} concurrent threads...")
        
        prompts = self.prompts[self.prompt_size]
        total_tokens = 0
        total_time = 0
        latencies = []
        prompt_response_data = []
        
        def process_request(i):
            prompt = prompts[i % len(prompts)]
            response, latency, token_count = self.call_ollama(prompt)
            
            prompt_response = {
                "prompt": prompt,
                "response": response.get('response', ''),
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0),
                "total_tokens": token_count
            }
            
            return token_count, latency, prompt_response
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(process_request, i) for i in range(self.num_requests)]
            for future in futures:
                tokens, latency, prompt_response = future.result()
                total_tokens += tokens
                latencies.append(latency)
                prompt_response_data.append(prompt_response)
                
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate throughput metrics
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        requests_per_second = self.num_requests / total_time if total_time > 0 else 0
        
        return {
            "tokens_per_second": tokens_per_second,
            "requests_per_second": requests_per_second,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "total_requests": self.num_requests,
            "concurrent_threads": self.num_threads,
            "latencies": latencies,
            "prompt_response_data": prompt_response_data
        }
    
    def measure_resource_usage(self) -> Dict[str, Any]:
        """
        Measure CPU and memory usage during benchmark.
        
        Returns:
            Dictionary with resource usage statistics
        """
        if not self.resource_usage["cpu"]:
            return {
                "error": "No resource usage data collected"
            }
        
        return {
            "cpu": {
                "min": min(self.resource_usage["cpu"]),
                "max": max(self.resource_usage["cpu"]),
                "mean": statistics.mean(self.resource_usage["cpu"]),
                "median": statistics.median(self.resource_usage["cpu"]),
            },
            "memory": {
                "min": min(self.resource_usage["memory"]),
                "max": max(self.resource_usage["memory"]),
                "mean": statistics.mean(self.resource_usage["memory"]),
                "median": statistics.median(self.resource_usage["memory"]),
            },
            "duration": self.resource_usage["end_time"] - self.resource_usage["start_time"],
            "samples": len(self.resource_usage["cpu"])
        }
    
    def measure_accuracy(self) -> Dict[str, Any]:
        """
        Measure response accuracy using predefined test cases.
        
        Returns:
            Dictionary with accuracy statistics
        """
        if not self.accuracy_test:
            return {"skipped": True}
        
        self._print(f"\n🎯 Measuring accuracy for {self.model_name}...")
        results = []
        
        for i, test in enumerate(self.accuracy_tests):
            question = test["question"]
            expected_keywords = test["expected_keywords"]
            
            self._print(f"  Testing question {i+1}/{len(self.accuracy_tests)}: {question}")
            response, _, token_count = self.call_ollama(question)
            answer_text = response.get('response', '').lower()
            
            # Check for expected keywords
            keywords_found = [keyword for keyword in expected_keywords if keyword.lower() in answer_text]
            
            result = {
                "question": question,
                "expected_keywords": expected_keywords,
                "keywords_found": keywords_found,
                "keyword_match_rate": len(keywords_found) / len(expected_keywords) if expected_keywords else 0,
                "answer": answer_text,
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0),
                "total_tokens": token_count
            }
            results.append(result)
        
        # Calculate overall accuracy
        if results:
            accuracy = sum(r["keyword_match_rate"] for r in results) / len(results)
        else:
            accuracy = 0
        
        return {
            "overall_accuracy": accuracy,
            "test_results": results
        }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Returns:
            Dictionary with all benchmark results
        """
        self._print(f"\n🚀 Starting LLM benchmark for model: {self.model_name}")
        self._print(f"Configuration: {self.num_requests} requests, {self.num_threads} threads, {self.prompt_size} prompts")
        
        # Start resource monitoring
        self._start_resource_monitoring()
        
        try:
            # Run latency tests
            latency_results = self.measure_latency()
            
            # Run throughput tests
            throughput_results = self.measure_throughput()
            
            # Run accuracy tests if enabled
            accuracy_results = self.measure_accuracy()
            
            # Compile results
            self.results = {
                "model": self.model_name,
                "config": {
                    "prompt_size": self.prompt_size,
                    "num_requests": self.num_requests,
                    "num_threads": self.num_threads,
                    "platform": "Raspberry Pi 5"
                },
                "latency": latency_results,
                "throughput": throughput_results,
                "accuracy": accuracy_results,
            }
        finally:
            # Stop resource monitoring
            self._stop_resource_monitoring()
            
            # Add resource usage data
            self.results["resource_usage"] = self.measure_resource_usage()
        
        # Add console output to results
        self._print_summary()
        self.results["console_output"] = "\n".join(self.console_output)
        
        # Save results if output file specified
        if self.output_file:
            with open(self.output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self._print(f"\n💾 Results saved to {self.output_file}")
        
        return self.results
    
    def _print_summary(self):
        """Print a summary of benchmark results."""
        self._print("\n📊 BENCHMARK SUMMARY 📊")
        self._print(f"Model: {self.model_name}")
        
        # Latency summary
        self._print("\nLatency:")
        for size, data in self.results["latency"].items():
            self._print(f"  {size.capitalize()}: {data['mean']:.2f}s (min: {data['min']:.2f}s, max: {data['max']:.2f}s)")
        
        # Throughput summary
        throughput = self.results["throughput"]
        self._print("\nThroughput:")
        self._print(f"  Tokens/sec: {throughput['tokens_per_second']:.2f}")
        self._print(f"  Requests/sec: {throughput['requests_per_second']:.2f}")
        self._print(f"  Total tokens: {throughput['total_tokens']}")
        
        # Resource usage summary
        ru = self.results["resource_usage"]
        self._print("\nResource Usage:")
        self._print(f"  CPU: {ru['cpu']['mean']:.1f}% (peak: {ru['cpu']['max']:.1f}%)")
        self._print(f"  Memory: {ru['memory']['mean']:.1f}% (peak: {ru['memory']['max']:.1f}%)")
        
        # Accuracy summary if available
        if not self.results["accuracy"].get("skipped", False):
            acc = self.results["accuracy"]
            self._print("\nAccuracy:")
            self._print(f"  Overall: {acc['overall_accuracy']*100:.1f}%")
            
            # Calculate average token counts
            prompt_tokens = [r["prompt_tokens"] for r in acc["test_results"]]
            completion_tokens = [r["completion_tokens"] for r in acc["test_results"]]
            total_tokens = [r["total_tokens"] for r in acc["test_results"]]
            
            if prompt_tokens and completion_tokens:
                self._print(f"  Avg. prompt tokens: {statistics.mean(prompt_tokens):.1f}")
                self._print(f"  Avg. completion tokens: {statistics.mean(completion_tokens):.1f}")
                self._print(f"  Avg. total tokens: {statistics.mean(total_tokens):.1f}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs using Ollama on Raspberry Pi 5")
    parser.add_argument("--model", "-m", type=str, required=True, 
                        help="Name of the Ollama model to benchmark")
    parser.add_argument("--requests", "-r", type=int, default=10, 
                        help="Number of requests to send")
    parser.add_argument("--threads", "-t", type=int, default=1, 
                        help="Number of concurrent threads for throughput testing")
    parser.add_argument("--prompt-size", "-p", type=str, choices=["small", "medium", "large"], 
                        default="medium", help="Size of prompts to use")
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="Output file to save benchmark results (JSON)")
    parser.add_argument("--accuracy", "-a", action="store_true", 
                        help="Run accuracy tests")
    
    args = parser.parse_args()
    
    # Check if Ollama is running
    try:
        response = requests.get(f"{OLLAMA_API}/tags")
        if response.status_code != 200:
            print(f"⚠️ Ollama API returned status code {response.status_code}. Make sure Ollama is running.")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama. Make sure it's running on http://localhost:11434")
        return
    
    # Run benchmark
    benchmark = OllamaBenchmark(
        model_name=args.model,
        num_threads=args.threads,
        num_requests=args.requests,
        prompt_size=args.prompt_size,
        output_file=args.output,
        accuracy_test=args.accuracy
    )
    
    benchmark.run_benchmark()

if __name__ == "__main__":
    main() 