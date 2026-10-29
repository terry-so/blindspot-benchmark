# Run local edits only

Generate images for Qwen-Image-2.1, FLUX.3 Klein, and FLUX.2 dev.
No judging or Azure credentials are required.

Use Linux, Python 3.11+, and a BF16-capable NVIDIA GPU.
The commands below use `run_local.py --skip-judge` for both the smoke test and full run.

1. Extract `data.zip` in the project root so `data/benchmark_cases` exists.
   Run all commands below from the project root.

2. Install CUDA-compatible PyTorch, then install dependencies and log in:

   ```bash
   python -m pip install -r environment/local_requirements.txt
   hf auth login
   ```

   Accept any required Hugging Face access terms for the three models and the
   `black-forest-labs/flux-3-image-klein-pup` prompt upsampler.

3. Verify the GPU:

   ```bash
   nvidia-smi
   python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported())"
   ```

   Both Python checks must print `True`.

4. Test one case at all levels for each model:

   ```bash
   for model in qwen-image-2.1 flux-3-klein flux-2-dev; do
     python run_local.py --model "$model" --skip-judge --cpu-offload \
       --levels L0 L1 L2a L2b L3-reset --smoke-test \
       --output "runs/local-generation-smoke/$model" || break
   done
   ```

   Check that all three models save edited PNGs and result JSON files with
   `status: "generated"`. If any model fails, stop and share its error.

5. Run all cases, once per level:

   ```bash
   for model in qwen-image-2.1 flux-3-klein flux-2-dev; do
     python run_local.py --model "$model" --skip-judge --cpu-offload \
       --levels L0 L1 L2a L2b L3-reset \
       --output "runs/local-generation/$model" || break
   done
   ```

Models run sequentially. L3 uses every step stored in each case, with each
step editing the previous output. It stops if image generation fails.
Refusal, fidelity, and realism scores remain unset.

Rerun the same command to resume. Use a new output folder if settings change.
Please return the complete `runs/local-generation` folder, including original
images, intermediate L3 images, result metadata, settings, and package versions.
