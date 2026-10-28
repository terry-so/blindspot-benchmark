1. Copy the project and the **`data/benchmark_cases` folder** to the GPU machine.

2. Open a terminal in the project folder. Install CUDA-compatible PyTorch, then the remaining dependencies:

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
   python run.py --config configs/benchmark.yaml --models flux-2-dev --levels L1 --smoke-test --output runs/test-flux2
   ```

6. Run the benchmark:

   ```bash
   python run.py --config configs/benchmark.yaml --models flux-2-dev --output runs/flux2
   ```

Replace `flux-2-dev` with `qwen-image-2.1` or `flux-3-klein` to run another local model.

Results are saved in the chosen output folder. Rerun the same command to resume.