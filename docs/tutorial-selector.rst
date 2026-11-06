.. meta::
   :description: Tutorial selector for the AI Developer Hub
   :keywords: AI, ROCm, developers, tutorials, guides, filters, selector, prerequisites, difficulty level, minimum GPU

.. _matrix:

******************************************
AI Developer Hub tutorial selector
******************************************

Select a knowledge level, tutorial type, and minimum GPU to filter the      
tutorials below. The minimum GPU is the least powerful GPU a tutorial requires; more     
powerful AMD GPUs will also work.      


.. selector:: Knowledge level
   :key: level

   .. selector-option:: All
      :value: all
      :width: 3

   .. selector-option:: Beginner
      :value: beginner
      :width: 3

   .. selector-option:: Intermediate
      :value: intermediate
      :width: 3

   .. selector-option:: Advanced
      :value: advanced
      :width: 3

.. selector:: Tutorial type
   :key: type

   .. selector-option:: All
      :value: all
      :width: 3

   .. selector-option:: Inference
      :value: inference
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

   .. selector-option:: Fine-tuning
      :value: fine-tuning
      :width: 2
      :show-cond: level=all level=intermediate level=beginner

   .. selector-option:: Pretraining
      :value: pretraining
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

   .. selector-option:: GPU Dev & Optimization
      :value: gpu-dev-opt
      :width: 3
      :show-cond: level=all level=beginner level=intermediate level=advanced

.. selector-dropdown:: Minimum GPU
   :key: gpu
   :show-cond: type=all

   .. selector-option:: All
      :value: all
      :width: 1

   .. selector-option:: AMD Radeon™ RX 7900 XTX (gfx1100)
      :value: rx7900xtx
      :width: 2
      :show-cond: level=all level=beginner

   .. selector-option:: AMD Radeon™ PRO W7900 (gfx1100)
      :value: w7900
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

   .. selector-option:: AMD Instinct™ MI210 (gfx90a)
      :value: mi210
      :width: 2
      :show-cond: level=all level=beginner

   .. selector-option:: AMD Instinct™ MI250X (gfx90a)
      :value: mi250x
      :width: 2
      :show-cond: level=all level=intermediate

   .. selector-option:: AMD Instinct™ MI300X (gfx942)
      :value: mi300x
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

   .. selector-option:: AMD Instinct™ MI350X (gfx950)
      :value: mi350x
      :width: 2
      :show-cond: level=all level=advanced

   .. selector-option:: AMD Instinct™ MI355X (gfx950)
      :value: mi355x
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

.. selector-dropdown:: Minimum GPU
   :key: gpu
   :show-cond: type=inference

   .. selector-option:: All
      :value: all
      :width: 1

   .. selector-option:: AMD Radeon™ RX 7900 XTX (gfx1100)
      :value: rx7900xtx
      :width: 2
      :show-cond: level=all level=beginner

   .. selector-option:: AMD Radeon™ PRO W7900 (gfx1100)
      :value: w7900
      :width: 2
      :show-cond: level=all level=beginner

   .. selector-option:: AMD Instinct™ MI300X (gfx942)
      :value: mi300x
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

.. selector-dropdown:: Minimum GPU
   :key: gpu
   :show-cond: type=fine-tuning

   .. selector-option:: All
      :value: all
      :width: 1

   .. selector-option:: AMD Instinct™ MI210 (gfx90a)
      :value: mi210
      :width: 2
      :show-cond: level=all level=beginner

   .. selector-option:: AMD Instinct™ MI300X (gfx942)
      :value: mi300x
      :width: 2
      :show-cond: level=all level=intermediate

.. selector-dropdown:: Minimum GPU
   :key: gpu
   :show-cond: type=pretraining

   .. selector-option:: All
      :value: all
      :width: 1

   .. selector-option:: AMD Instinct™ MI250X (gfx90a)
      :value: mi250x
      :width: 2
      :show-cond: level=all level=intermediate

   .. selector-option:: AMD Instinct™ MI300X (gfx942)
      :value: mi300x
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

.. selector-dropdown:: Minimum GPU
   :key: gpu
   :show-cond: type=gpu-dev-opt

   .. selector-option:: All
      :value: all
      :width: 1

   .. selector-option:: AMD Radeon™ PRO W7900 (gfx1100)
      :value: w7900
      :width: 2
      :show-cond: level=all level=intermediate

   .. selector-option:: AMD Instinct™ MI300X (gfx942)
      :value: mi300x
      :width: 2
      :show-cond: level=all level=intermediate

   .. selector-option:: AMD Instinct™ MI350X (gfx950)
      :value: mi350x
      :width: 2
      :show-cond: level=all level=advanced

   .. selector-option:: AMD Instinct™ MI355X (gfx950)
      :value: mi355x
      :width: 2
      :show-cond: level=all level=beginner level=intermediate

----

Here are the selected tutorials.

.. selected:: level=all type=all gpu=all

    **Inference tutorials**

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
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
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

    **Fine-tuning tutorials**

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`
    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

    **Pretraining tutorials**

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

    **GPU development and optimization tutorials**

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`
    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`
    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=all type=all gpu=rx7900xtx

    **Inference tutorials**

    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`

.. selected:: level=all type=all gpu=w7900

    **Inference tutorials**

    * :doc:`RAG with LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`
    * :doc:`Building AI pipelines for voice assistants <./notebooks/inference/voice_pipeline_rag_ollama>`

    **GPU development and optimization tutorials**

    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`

.. selected:: level=all type=all gpu=mi250x

    **Pretraining tutorials**

    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

.. selected:: level=all type=all gpu=mi210

    **Fine-tuning tutorials**

    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

.. selected:: level=all type=all gpu=mi300x

    **Inference tutorials**

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
    * :doc:`DeepSeek Janus Pro on CPU or GPU <./notebooks/inference/deepseek_janus_cpu_gpu>`
    * :doc:`DeepSeek-R1 with vLLM V1 <./notebooks/inference/vllm_v1_DSR1>`
    * :doc:`AI agent with MCPs using vLLM and PydanticAI <./notebooks/inference/build_airbnb_agent_mcp>`
    * :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Deploying with vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`From chatbot to rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`OCR with vision-language models with vLLM <./notebooks/inference/ocr_vllm>`
    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

    **Fine-tuning tutorials**

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

    **Pretraining tutorials**

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

    **GPU development and optimization tutorials**

    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`

.. selected:: level=all type=all gpu=mi350x

    **GPU development and optimization tutorials**

    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=all type=all gpu=mi355x

    **GPU development and optimization tutorials**

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

.. selected:: level=all type=inference gpu=all

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
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
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

.. selected:: level=all type=inference gpu=rx7900xtx

    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`

.. selected:: level=all type=inference gpu=w7900

    * :doc:`RAG with LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`
    * :doc:`Building AI pipelines for voice assistants <./notebooks/inference/voice_pipeline_rag_ollama>`

.. selected:: level=all type=inference gpu=mi300x

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
    * :doc:`DeepSeek Janus Pro on CPU or GPU <./notebooks/inference/deepseek_janus_cpu_gpu>`
    * :doc:`DeepSeek-R1 with vLLM V1 <./notebooks/inference/vllm_v1_DSR1>`
    * :doc:`AI agent with MCPs using vLLM and PydanticAI <./notebooks/inference/build_airbnb_agent_mcp>`
    * :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Deploying with vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`From chatbot to rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`OCR with vision-language models with vLLM <./notebooks/inference/ocr_vllm>`
    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

.. selected:: level=all type=fine-tuning gpu=all

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`
    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

.. selected:: level=all type=fine-tuning gpu=mi210

    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

.. selected:: level=all type=fine-tuning gpu=mi300x

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

.. selected:: level=all type=pretraining gpu=all

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

.. selected:: level=all type=pretraining gpu=mi250x

    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

.. selected:: level=all type=pretraining gpu=mi300x

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

.. selected:: level=all type=gpu-dev-opt gpu=all

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`
    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`
    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=all type=gpu-dev-opt gpu=w7900

    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`

.. selected:: level=all type=gpu-dev-opt gpu=mi300x

    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`

.. selected:: level=all type=gpu-dev-opt gpu=mi350x

    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=all type=gpu-dev-opt gpu=mi355x

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

.. selected:: level=beginner type=all gpu=all

    **Inference tutorials**

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
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`

    **Pretraining tutorials**

    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

    **Fine-tuning tutorials**

    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

    **GPU development and optimization tutorials**

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`

.. selected:: level=beginner type=inference gpu=all

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
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`

.. selected:: level=beginner type=all gpu=rx7900xtx

    **Inference tutorials**

    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`

.. selected:: level=beginner type=all gpu=w7900

    **Inference tutorials**

    * :doc:`RAG with LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`
    * :doc:`Building AI pipelines for voice assistants <./notebooks/inference/voice_pipeline_rag_ollama>`

.. selected:: level=beginner type=all gpu=mi300x

    **Inference tutorials**

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`DeepSeek Janus Pro on CPU or GPU <./notebooks/inference/deepseek_janus_cpu_gpu>`
    * :doc:`DeepSeek-R1 with vLLM V1 <./notebooks/inference/vllm_v1_DSR1>`
    * :doc:`AI agent with MCPs using vLLM and PydanticAI <./notebooks/inference/build_airbnb_agent_mcp>`
    * :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Deploying with vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`From chatbot to rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`OCR with vision-language models with vLLM <./notebooks/inference/ocr_vllm>`
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`

    **Pretraining tutorials**

    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

.. selected:: level=beginner type=all gpu=mi355x

    **GPU development and optimization tutorials**

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`

.. selected:: level=beginner type=all gpu=mi210

    **Fine-tuning tutorials**

    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

.. selected:: level=beginner type=fine-tuning gpu=all

    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

.. selected:: level=beginner type=fine-tuning gpu=mi210

    * :doc:`Training a MicroDuck walking policy with reinforcement learning <./notebooks/fine_tune/microduck_lab_standalone>`

.. selected:: level=beginner type=inference gpu=rx7900xtx

    * :doc:`Text-to-video generation with ComfyUI <./notebooks/inference/t2v_comfyui_radeon>`

.. selected:: level=beginner type=inference gpu=w7900

    * :doc:`RAG with LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`
    * :doc:`Building AI pipelines for voice assistants <./notebooks/inference/voice_pipeline_rag_ollama>`

.. selected:: level=beginner type=inference gpu=mi300x

    * :doc:`ChatQnA vLLM deployment and performance evaluation <./notebooks/inference/opea_deployment_and_evaluation>`
    * :doc:`DeepSeek Janus Pro on CPU or GPU <./notebooks/inference/deepseek_janus_cpu_gpu>`
    * :doc:`DeepSeek-R1 with vLLM V1 <./notebooks/inference/vllm_v1_DSR1>`
    * :doc:`AI agent with MCPs using vLLM and PydanticAI <./notebooks/inference/build_airbnb_agent_mcp>`
    * :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Deploying with vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`From chatbot to rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`OCR with vision-language models with vLLM <./notebooks/inference/ocr_vllm>`
    * :doc:`DeepSeek-R1 with SGLang <./notebooks/inference/deepseekr1_sglang>`
    * :doc:`Deploy OpenClaw with Qwen3.5 and vLLM <./notebooks/inference/openclaw_vllm>`

.. selected:: level=beginner type=pretraining gpu=all

    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

.. selected:: level=beginner type=pretraining gpu=mi300x

    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Training and serving a FinalNet CTR model with Triton Inference Server <./notebooks/pretrain/triton_inference_server_benchmark>`

.. selected:: level=beginner type=gpu-dev-opt gpu=all

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`

.. selected:: level=beginner type=gpu-dev-opt gpu=mi355x

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`

.. selected:: level=intermediate type=all gpu=all

    **Inference tutorials**

    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

    **Fine-tuning tutorials**

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

    **Pretraining tutorials**

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

    **GPU development and optimization tutorials**

    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

.. selected:: level=intermediate type=inference gpu=all

    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

.. selected:: level=intermediate type=all gpu=mi250x

    **Pretraining tutorials**

    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

.. selected:: level=intermediate type=all gpu=w7900

    **GPU development and optimization tutorials**

    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`

.. selected:: level=intermediate type=all gpu=mi300x

    **Inference tutorials**

    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

    **Fine-tuning tutorials**

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

    **Pretraining tutorials**

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`

    **GPU development and optimization tutorials**

    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`

.. selected:: level=intermediate type=all gpu=mi355x

    **GPU development and optimization tutorials**

    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

.. selected:: level=intermediate type=inference gpu=mi300x

    * :doc:`Speculative decoding with vLLM <./notebooks/inference/speculative_decoding_deep_dive>`
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`
    * :doc:`Running ComfyUI generative workflows from Python on AMD Instinct GPUs <./notebooks/inference/t2v_comfyui_api_mode_instinct>`
    * :doc:`Multi-agent incident triage with OpenClaw on AMD Instinct <./notebooks/inference/graph_multiagent_openclaw>`
    * :doc:`Profiling and optimizing an AI agent on AMD Instinct GPUs <./notebooks/inference/agentic_ai_profiling>`

.. selected:: level=intermediate type=fine-tuning gpu=all

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

.. selected:: level=intermediate type=fine-tuning gpu=mi300x

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

.. selected:: level=intermediate type=pretraining gpu=all

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

.. selected:: level=intermediate type=pretraining gpu=mi250x

    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

.. selected:: level=intermediate type=pretraining gpu=mi300x

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`

.. selected:: level=intermediate type=gpu-dev-opt gpu=all

    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

.. selected:: level=intermediate type=gpu-dev-opt gpu=w7900

    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`

.. selected:: level=intermediate type=gpu-dev-opt gpu=mi300x

    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`

.. selected:: level=intermediate type=gpu-dev-opt gpu=mi355x

    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

.. selected:: level=advanced type=all gpu=all

    **GPU development and optimization tutorials**

    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=advanced type=gpu-dev-opt gpu=all

    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=advanced type=all gpu=mi350x

    **GPU development and optimization tutorials**

    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`

.. selected:: level=advanced type=gpu-dev-opt gpu=mi350x

    * :doc:`FP8 GEMM optimization on AMD CDNA4-based GPUs <./notebooks/gpu_dev_optimize/fp8_gemm_hip_cdna4>`
