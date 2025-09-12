.. meta::
   :description: The AI Developer Hub provides tutorials and guides for training, fine-tuning, and inference
   :keywords: AI, ROCm, developers, tutorials, guides, training, fine-tuning, inference

.. _index:

****************************************
Tutorials for AI developers
****************************************

The AI Developer Hub contains AMD ROCm tutorials in Jupyter Notebook format for training, fine-tuning, and inference.
It leverages popular machine learning frameworks on AMD GPUs.

These tutorials are organized into four main categories:

*  **Inference**: Resources for running inference with trained models.
*  **Fine-tuning**: Examples and guides for fine-tuning machine learning models.
*  **Pretraining**: Tutorials on pretraining models from scratch.
*  **GPU development and optimization**: Resources for optimizing AI compute and kernel development on GPUs.

All tutorials on the AI Developer Hub are available to download in the Jupyter Notebook format from the
public GitHub repository at `<https://github.com/ROCm/gpuaidev>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Inference tutorials

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`
    * :doc:`DeepSeek Janus Pro on CPU or GPU <./notebooks/inference/deepseek_janus_cpu_gpu>`
    * :doc:`DeepSeek-R1 with vLLM V1 <./notebooks/inference/vllm_v1_DSR1>`
    * :doc:`AI agent with MCPs using vLLM and PydanticAI <./notebooks/inference/build_airbnb_agent_mcp>`
    * :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Deploying with vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`From chatbot to rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`RAG with LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`
    * :doc:`OCR with vision-language models with vLLM <./notebooks/inference/ocr_vllm>`
    * :doc:`Building AI pipelines for voice assistants <./notebooks/inference/voice_pipeline_rag_ollama>`
    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`

  .. grid-item-card:: Fine-tuning tutorials

    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`

  .. grid-item-card:: Pretraining tutorials

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`

  .. grid-item-card:: GPU development and optimization tutorials

    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

To contribute to the documentation, see
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find ROCm licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
