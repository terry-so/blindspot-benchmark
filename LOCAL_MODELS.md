1. Place **`data` folder** in the project root. 

2. Install CUDA-compatible PyTorch, then the remaining dependencies:

   ```bash
   pip install -r environment/local_requirements.txt
   hf auth login
   ```

   Accept the selected model’s access terms on Hugging Face if required.

3. Set the Azure judge credentials:

   ```bash
   export AZURE_OPENAI_BASE_URL="https://YOUR-ENDPOINT/openai/v1/"
   export Azure_API_KEY="YOUR-KEY"
   ```

4. In `configs/benchmark.yaml`, set `judge.model` to the Azure judge deployment name.


5. Test one image:

   ```bash
   python run.py --config configs/benchmark.yaml \
     --models qwen-image-2.1 flux-3-klein flux-2-dev \
     --levels L1 --smoke-test \
     --output runs/local-smoke-test
   ```

   It should output results in `runs/local-smoke-test`.

6. Run the benchmark:

   ```bash
   python run.py --config configs/benchmark.yaml \
     --models qwen-image-2.1 flux-3-klein flux-2-dev \
     --output runs/local-benchmark
   ```

Replace `flux-2-dev`, `qwen-image-2.1` or `flux-3-klein` to run another local model.

Results are saved in the chosen output folder. Rerun the same command to resume.

**Hardware requirement** (According to AI)

| Component | target |
|---|---|
| GPU | One A100 80 GB or H100 80 GB |
| System RAM | 128 GB minimum planning target; 192–256 GB preferred |
| CPU | 8–16 cores |
| Free disk space | 500 GB for all model downloads, caches, and outputs |
| OS | Linux with a working NVIDIA driver and CUDA-enabled PyTorch |
| Configuration | `cpu_offload: true` |