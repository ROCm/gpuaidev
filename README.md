# AI Developer Hub

Welcome to the **AI Developer Hub** repository! This project contains Jupyter Notebook tutorials and guides for training, fine-tuning, and inference using popular machine learning frameworks on AMD GPUs.

## Current Notebooks

| Category                 | Title                                             | GitHub Link                                                                                                      | AMD Tutorial Page                                                                                                                       |
|--------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Inference Tutorials**  | Hugging Face Transformers                         | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/1_inference_ver3_HF_transformers.ipynb) | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/1_inference_ver3_HF_transformers.html) |
|                          | Hugging Face TGI                                  | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/2_inference_ver3_HF_TGI.ipynb)          | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/2_inference_ver3_HF_TGI.html)          |
|                          | Deploying with vLLM                               | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/3_inference_ver3_HF_vllm.ipynb)         | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/3_inference_ver3_HF_vllm.html)         |
|                          | From Chatbot to Rap Bot with vLLM                | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/rapbot_vllm.ipynb)                      | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/rapbot_vllm.html)                      |
|                          | RAG with LlamaIndex and Ollama                   | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/rag_ollama_llamaindex.ipynb)            | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/rag_ollama_llamaindex.html)            |
|                          | OCR with Vision-Language Models with vLLM        | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/ocr_vllm.ipynb)                         | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/ocr_vllm.html)                         |
|                          | Building AI Pipelines for Voice Assistants       | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/inference/voice_pipeline_rag_ollama.ipynb)         | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/inference/voice_pipeline_rag_ollama.html)         |
| **Fine-Tuning Tutorials**| VLM with PEFT                                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/fine_tuning_lora_qwen2vl.ipynb)          | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/fine_tuning_lora_qwen2vl.html)          |
|                          | LLM with LoRA                                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/LoRA_Llama-3.2.ipynb)                    | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/LoRA_Llama-3.2.html)                    |
|                          | LLM with QLoRA                                    | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/QLoRA_Llama-3.1.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/QLoRA_Llama-3.1.html)                   |
|                          | Llama-3.1 8B with torchtune                       | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/fine_tune/torchtune_llama3.ipynb)                  | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/fine_tune/torchtune_llama3.html)                  |
| **Pretraining Tutorials**| OLMo Model with PyTorch FSDP                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/torch_fsdp.ipynb)                          | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torch_fsdp.html)                          |
|                          | Training Configuration with Megatron-LM          | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/setup_tutorial.ipynb)                      | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/setup_tutorial.html)                      |
|                          | LLM with Megatron-LM                             | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/train_llama_mock_data.ipynb)               | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/train_llama_mock_data.html)               |
|                          | Llama-3.1 8B with torchtitan                     | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/pretrain/torchtitan_llama3.ipynb)                   | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/pretrain/torchtitan_llama3.html)                   |
| **About**                | Licensing and Support Information                | [GitHub](https://github.com/ROCm/gpuaidev/blob/main/docs/notebooks/licensing.md)                                       | [AMD Tutorial](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/notebooks/licensing.html)                                    |

## Repository Structure

The tutorials are organized into three main categories:

- **Fine-Tuning**: Examples and guides for fine-tuning machine learning models.
- **Pretraining**: Tutorials on pretraining models from scratch.
- **Inference**: Resources for running inference with trained models.

### Directory Layout

```
github_repo/
├── docs/                          # Documentation for the tutorials
│   ├── index.md                   # Main documentation index
│   ├── fine_tune.md               # Fine-tuning tutorials index
│   ├── pretrain.md                # Pretraining tutorials index
│   ├── inference.md               # Inference tutorials index
│   └── notebooks/                 # Jupyter notebooks organized by category
│       ├── fine_tune/             # Fine-tuning notebooks
│       ├── pretrain/              # Pretraining notebooks
│       └── inference/             # Inference notebooks
```
