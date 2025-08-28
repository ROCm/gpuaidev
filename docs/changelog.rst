.. meta::
   :description: Changelog for the AI Developer Hub
   :keywords: AI, ROCm, developers, tutorials, guides, changelog

.. _changelog:

****************************************
Changelog for the AI Developer Hub
****************************************

AI developer tutorials are available at the :doc:`AI Developer Hub <./index>`.


Version 5.1
===========

Added
------

*  New inference tutorials:

   * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`

Version 5.0
===========

Added
------

*  New inference tutorials:

   * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
   * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`
   * :doc:`DeepSeek Janus Pro on CPU or GPU <./notebooks/inference/deepseek_janus_cpu_gpu>`
   * :doc:`DeepSeek-R1 with vLLM V1 <./notebooks/inference/vllm_v1_DSR1>`

*  New GPU development and optimization tutorial:

   * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`

Version 4.0
===========

Added
------

*  New inference tutorial:

   * :doc:`AI agent with MCPs using vLLM and PydanticAI <./notebooks/inference/build_airbnb_agent_mcp>`

*  New GPU development and optimization tutorials:

   * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
   * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
   * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

Changed
-------

*  DDIM pretraining tutorial renamed to :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
   with some minor changes.
  
Version 3.1
===========

Added
------

*  New fine-tuning tutorial:

   * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`

Version 3.0
===========

Added
------

*  New inference tutorials:

   *  :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
   *  :doc:`Llama Stack <./notebooks/inference/llama-stack-rocm>`
   *  :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`


*  New fine-tuning tutorial:
  
   *  :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`

*  New pretraining tutorial:

   *  :doc:`DDIM  <./notebooks/pretrain/ddim_pretrain>`

Version 2.0
===========

Added
------

*  New inference tutorials:

   *  :doc:`OCR with vision-language models with vLLM <./notebooks/inference/ocr_vllm>`
   *  :doc:`Building AI pipelines for voice assistants <./notebooks/inference/voice_pipeline_rag_ollama>`

*  New fine-tuning tutorial:
  
   *  :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`

*  New pretraining tutorial:

   *  :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`

Changed
-------

*  Updated the supported ROCm versions for some guides
*  Additional minor changes

Version 1.0
===========

Added
------

*  New inference tutorials:

   *  :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
   *  :doc:`Hugging Face TGI <./notebooks/inference/2_inference_ver3_HF_TGI>`
   *  :doc:`Deploying with vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
   *  :doc:`From chatbot to rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
   *  :doc:`RAG with LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`

*  New fine-tuning tutorials:
  
   *  :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
   *  :doc:`LLM with LoRA <./notebooks/fine_tune/LoRA_Llama-3.2>`
   *  :doc:`LLM with QLoRA <./notebooks/fine_tune/QLoRA_Llama-3.1>`

*  New pretraining tutorials:

   *  :doc:`OLMo model with PyTorch FSDP <./notebooks/pretrain/torch_fsdp>`
   *  :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
   *  :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`