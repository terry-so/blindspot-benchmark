Run these commands from the project root using Python 3.11+ on Linux.
GPU inference has not yet been validated on the target hardware; run the smoke test first.

1. Extract `data.zip` in the project root. The resulting path must be `data/benchmark_cases`.

2. Install CUDA-compatible PyTorch, then the remaining dependencies:

   ```bash
   python -m pip install -r environment/local_requirements.txt
   hf auth login
   ```

   Accept the selected model’s access terms on Hugging Face if required.

   Verify GPU access before continuing:

   ```bash
   nvidia-smi
   python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
   ```

   Both Python checks must print `True`.

3. Set the Azure judge credentials:

   ```bash
   export AZURE_OPENAI_BASE_URL="https://YOUR-ENDPOINT/openai/v1/"
   export Azure_API_KEY="YOUR-KEY"
   ```

4. In `configs/benchmark.yaml`, set `judge.model` to the Azure judge deployment name.
   The judge must support image inputs. If the endpoint restricts IP addresses,
   allow the GPU machine's outbound public IP. Local runs currently require this judge.

5. Test one image:

   ```bash
   python run.py --config configs/benchmark.yaml \
     --models qwen-image-2.1 flux-3-klein flux-2-dev \
     --levels L1 --smoke-test \
     --output runs/local-smoke-test
   ```

   This runs one L1 case per model, sequentially, including Azure judging.
   Continue only if all three models save an edited image and a result JSON with `status: "ok"`.

6. Run the benchmark:

   ```bash
   python run.py --config configs/benchmark.yaml \
     --models qwen-image-2.1 flux-3-klein flux-2-dev \
     --output runs/local-benchmark
   ```

This uses all dataset cases and the levels/repeats in `configs/benchmark.yaml`.
To run one model, keep only its name after `--models`.

Return the complete `runs/local-benchmark` folder, including originals, edited images,
scores, settings, and recorded package versions. Rerun the same command to resume.

Suggested hardware for the current BF16 implementation with `cpu_offload: true`:

| Component | target |
|---|---|
| GPU | One A100 80 GB or H100 80 GB |
| System RAM | 128 GB minimum planning target; 192–256 GB preferred |
| CPU | 8–16 cores |
| Free disk space | 500 GB for all model downloads, caches, and outputs |
| OS | Linux with a working NVIDIA driver and CUDA-enabled PyTorch |
| Configuration | `cpu_offload: true` |

These are planning estimates, not measured minimums for this benchmark.
