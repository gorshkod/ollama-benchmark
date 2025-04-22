# LLM Benchmark Tool for Raspberry Pi 5

This tool benchmarks Large Language Models (LLMs) running on Ollama on Raspberry Pi 5, measuring key performance metrics:

- **Latency**: Response time for different prompt sizes
- **Throughput**: Tokens per second and requests per second
- **Resource Utilization**: CPU and memory usage
- **Response Accuracy**: Using predefined test cases
- **Detailed Records**: Complete prompts, responses, and token counts
- **Console Output**: All console output is saved in the JSON results file

## Requirements

- Raspberry Pi 5
- Ollama installed and running
- Python 3.7+
- Required Python packages (install with `pip install -r requirements.txt`)

## Installation

1. Clone this repository or copy the files to your Raspberry Pi 5
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure Ollama is installed and running on your Raspberry Pi 5
   - Install Ollama following the instructions at [ollama.ai](https://ollama.ai)
   - Start Ollama with: `ollama serve`
   - Pull your desired model: `ollama pull <model_name>`

## Usage

```bash
python benchmark_llm.py --model <model_name> [options]
```

### Command Line Options

- `--model, -m`: Name of the Ollama model to benchmark (required)
- `--requests, -r`: Number of requests to send for each test (default: 10)
- `--threads, -t`: Number of concurrent threads for throughput testing (default: 1)
- `--prompt-size, -p`: Size of prompts to use: small, medium, or large (default: medium)
- `--output, -o`: Output file to save benchmark results in JSON format
- `--accuracy, -a`: Run accuracy tests

### Examples

Basic benchmark with default settings:
```bash
python benchmark_llm.py --model llama2
```

Comprehensive benchmark with all tests:
```bash
python benchmark_llm.py --model llama2 --requests 20 --threads 2 --prompt-size medium --accuracy --output results.json
```

Testing multiple models:
```bash
for model in "llama2" "mistral" "phi"; do
  python benchmark_llm.py --model $model --output "${model}_results.json" --accuracy
done
```

## Output

The tool provides a summary of the benchmark results on the console and can save detailed results in JSON format for further analysis. The saved JSON includes complete records of:

- All prompts and responses
- Token counts (prompt tokens, completion tokens, and total)
- Detailed performance metrics
- Complete console output text

### Sample Output

```
📊 BENCHMARK SUMMARY 📊
Model: llama2

Latency:
  Medium: 2.45s (min: 2.12s, max: 3.01s)

Throughput:
  Tokens/sec: 25.64
  Requests/sec: 0.42
  Total tokens: 3245

Resource Usage:
  CPU: 78.5% (peak: 95.2%)
  Memory: 45.3% (peak: 52.1%)

Accuracy:
  Overall: 92.5%
  Avg. prompt tokens: 12.4
  Avg. completion tokens: 156.8
  Avg. total tokens: 169.2
```

## Tips for Raspberry Pi 5

- Ensure adequate cooling for your Raspberry Pi 5 during benchmarks
- Close other applications to minimize resource contention
- For better performance, consider using a fast microSD card or SSD
- Monitor temperature with `vcgencmd measure_temp` 