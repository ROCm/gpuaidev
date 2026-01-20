# AI Developer Hub

Welcome to the **AI Developer Hub** repository! This project contains Jupyter Notebook tutorials and guides for training, fine-tuning, and inference using popular machine learning frameworks on AMD GPUs.

## Current Notebooks

| Category                 | Title                                             | GitHub Link                                                                                                      | AMD Tutorial Page                                                                                                                       |
|--------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Inference Tutorials**  | ChatQnA vLLM deployment and performance evaluation| [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/opea_deployment_and_evaluation.ipynb) | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/opea_deployment_and_evaluation.html) |
|                          | Text-to-video generation with ComfyUI on Radeon GPU  | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/t2v_comfyui_radeon.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/t2v_comfyui_radeon.html)         |
|                          | DeepSeek Janus Pro on CPU or GPU  | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/deepseek_janus_cpu_gpu.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/deepseek_janus_cpu_gpu.html)         |
|                          | DeepSeek-R1 with vLLM V1 | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/vllm_v1_DSR1.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/vllm_v1_DSR1.html)         |
|                          | AI agent with MCPs using vLLM and PydanticAI      | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/build_airbnb_agent_mcp.ipynb)       | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/build_airbnb_agent_mcp.html) |
|                          | Multi-agents with Google ADK and A2A protocol     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.ipynb)       | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs.html) |
|                          | Hugging Face Transformers                         | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/1_inference_ver3_HF_transformers.ipynb) | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/1_inference_ver3_HF_transformers.html) |
|                          | Deploying with vLLM                               | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/3_inference_ver3_HF_vllm.ipynb)         | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/3_inference_ver3_HF_vllm.html)         |
|                          | From chatbot to rap bot with vLLM                | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/rapbot_vllm.ipynb)                      | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/rapbot_vllm.html)                      |
|                          | RAG with LlamaIndex and Ollama                   | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/rag_ollama_llamaindex.ipynb)            | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/rag_ollama_llamaindex.html)            |
|                          | OCR with vision-language models with vLLM        | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/ocr_vllm.ipynb)                         | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html)                         |
|                          | Building AI pipelines for voice assistants       | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/voice_pipeline_rag_ollama.ipynb)         | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/voice_pipeline_rag_ollama.html)         |
|                          | Speculative decoding with vLLM                   | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/speculative_decoding_deep_dive.ipynb)    | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/speculative_decoding_deep_dive.html)         |
|                          | DeepSeekR1 with SGLang and example applications  | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/deepseekr1_sglang.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/deepseekr1_sglang.html)         |
|                          | PD disaggregation with SGLang  | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU.html)         |
|                          | Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/mtp.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/mtp.html)         |
| **Fine-Tuning Tutorials**| Customize Qwen-Image with DiffSynth-Studio                                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/qwen_image.ipynb)          | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/qwen_image.html)          |
|                          | VLM with PEFT                                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/fine_tuning_lora_qwen2vl.ipynb)          | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/fine_tuning_lora_qwen2vl.html)          |
|                          | Llama-3.1 8B with torchtune                       | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/torchtune_llama3.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/torchtune_llama3.html)                  |
|                          | Llama-3.1 8B with Llama Factory                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/llama_factory_llama3.ipynb)             | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/llama_factory_llama3.html)                   |
|                          | GRPO with Unsloth                    | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO.ipynb)             | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO.html)                  |
|                          | GRPO with slime                    | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/slime_qwen3_4B_GRPO.ipynb)             | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/slime_qwen3_4B_GRPO.html)                  |              
| **Pretraining Tutorials**| Training configuration with Megatron-LM          | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/setup_tutorial.ipynb)                      | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)                      |
|                          | LLM with Megatron-LM                             | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/train_llama_mock_data.ipynb)               | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html)               |
|                          | Llama-3.1 8B with torchtitan                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/torchtitan_llama3.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_llama3.html)                   |
|                          | Draft model training with SGLang SpecForge       | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/SpecForge_SGlang.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/SpecForge_SGlang.html)                   |
|                          | Custom diffusion model                 | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/ddim_pretrain.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/ddim_pretrain.html)                   |
|                          | Pretraining with TorchTitan                      | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/torchtitan_deepseek.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_deepseek.html)                   |
|                          | Training a model with Primus                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/training_with_primus.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/training_with_primus.html)                   |
| **GPU Development and Optimization Tutorials**| Quark MXFP4 quantization for vLLM  | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm.html)      |
|                          | GPU kernel development and assessment with Helion   |[GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/gpu_dev_optimize/helion_gpu_kernel_dev.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/helion_gpu_kernel_dev.html)    |
|                          | MLA decoding kernel of AITER library    | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/aiter_mla_decode_kernel.html)         |
|                          | Kernel development and optimization with Triton                                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/gpu_dev_optimize/triton_kernel_dev.ipynb)          | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/triton_kernel_dev.html)          |
|                          | Profiling Llama-4 inference with vLLM            | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/gpu_dev_optimize/llama4_profiling_vllm.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/llama4_profiling_vllm.html)                   |
|                          | FP8 quantization with AMD Quark for vLLM    | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm.html)                   |
| **About**                | Licensing and Support Information                | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/licensing.md)                                       | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/licensing.html)                                    |



## Repository Structure

The tutorials are organized into four main categories:

- **Fine-Tuning**: Examples and guides for fine-tuning machine learning models.
- **Pretraining**: Tutorials on pretraining models from scratch.
- **Inference**: Resources for running inference with trained models.
- **GPU development and optimization**: Resources for optimizing AI compute and kernel development on GPUs.


### Directory Layout

```
github_repo/
├── docs/                          # Documentation for the tutorials
│   ├── index.md                   # Main documentation index
│   ├── fine_tune.md               # Fine-tuning tutorials index
│   ├── pretrain.md                # Pretraining tutorials index
│   ├── inference.md               # Inference tutorials index
│   └── notebooks/                 # Jupyter notebooks organized by category
│       ├── gpu_dev_optimize/      # GPU development and optimization notebooks
│       ├── fine_tune/             # Fine-tuning notebooks
│       ├── pretrain/              # Pretraining notebooks
│       └── inference/             # Inference notebooks
```
