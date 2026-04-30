.. meta::
   :description: The AI Developer Hub provides tutorials and guides for training, fine-tuning, and inference
   :keywords: AI, ROCm, developers, tutorials, guides, training, fine-tuning, inference

.. _index:

****************************************
Tutorials for AI developers
****************************************

The AI Developer Hub contains AMD ROCm tutorials in Jupyter Notebook format for training, fine-tuning, and inference.
It leverages popular machine learning frameworks on AMD GPUs.

.. admonition:: New tutorials
   
   * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`
   * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

These tutorials are organized into four main categories:

*  **Inference**: Resources for running inference with trained models.
*  **Fine-tuning**: Examples and guides for fine-tuning machine learning models.
*  **Pretraining**: Tutorials on pretraining models from scratch.
*  **GPU development and optimization**: Resources for optimizing AI compute and kernel development on GPUs.

All tutorials on the AI Developer Hub are available to download in the Jupyter Notebook format from the
public GitHub repository at `<https://github.com/ROCm/gpuaidev>`_.

.. tip::

   To determine which tutorials are best suited for your experience and knowledge level,
   see the :doc:`AI Developer Hub tutorial selector <./tutorial-selector>`.

Running the notebooks
=====================

Each tutorial is a Jupyter Notebook (``.ipynb`` file). To run a notebook:

1. Open the downloaded notebook in `JupyterLab <https://jupyterlab.readthedocs.io/en/latest/>`_ or `Jupyter Notebook <https://jupyter-notebook.readthedocs.io/en/latest/>`_.
2. Execute cells sequentially from top to bottom using **Shift + Enter** (run the current cell and move to the next) or **Ctrl + Enter** (run and stay on the current cell).

.. rubric:: Understanding cell status

Jupyter shows the execution status of each code cell on its left side:

- ``[ ]:`` — the cell has not been run yet.
- ``[*]:`` — the cell is **currently running**; wait for it to finish before moving on.
- ``[1]:``, ``[2]:``, … — the cell has **finished** (the number indicates the execution order).

A cell is considered complete when its indicator changes from ``[*]:`` to a number such as ``[1]:``.

.. note::

   There is no separate "mark as complete" button. Successfully executing a cell is how you
   complete it. Run all cells in order from top to bottom to complete the tutorial.

.. rubric:: Recommended workflow

1. Read each cell's description before running it.
2. Run cells **top to bottom** — skipping cells can cause errors.
3. If a cell shows an error, read the output message, fix the issue, and re-run the cell.
4. Long-running cells (such as model downloads or training loops) display a progress bar — wait for them to finish before continuing.
5. To restart a notebook from scratch, select **Kernel > Restart & Run All** in the Jupyter menu.

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
    * :doc:`PD disaggregation with SGLang <./notebooks/inference/SGlang_PD_Disagg_On_AMD_GPU>`
    * :doc:`Accelerating DeepSeek-V3 inference using multi-token prediction in SGLang <./notebooks/inference/mtp>`
    * :doc:`Multi-agents with Google ADK and A2A protocol <./notebooks/inference/power-Google-ADK-on-AMD-platform-and-local-LLMs>`

  .. grid-item-card:: Fine-tuning tutorials

    * :doc:`Customize Qwen-Image with DiffSynth-Studio <./notebooks/fine_tune/qwen_image>`
    * :doc:`VLM with PEFT <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.1 8B with torchtune <./notebooks/fine_tune/torchtune_llama3>`
    * :doc:`Llama-3.1 8B with Llama-Factory <./notebooks/fine_tune/llama_factory_llama3>`
    * :doc:`GRPO with Unsloth <./notebooks/fine_tune/unsloth_Llama3_1_8B_GRPO>`
    * :doc:`GRPO with slime <./notebooks/fine_tune/slime_qwen3_4B_GRPO>`

  .. grid-item-card:: Pretraining tutorials

    * :doc:`Training configuration with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`LLM with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`
    * :doc:`Llama-3.1 8B with torchtitan <./notebooks/pretrain/torchtitan_llama3>`
    * :doc:`Custom diffusion model with PyTorch <./notebooks/pretrain/ddim_pretrain>`
    * :doc:`Speculative decoding draft model with SpecForge <./notebooks/pretrain/SpecForge_SGlang>`
    * :doc:`Pretraining with TorchTitan <./notebooks/pretrain/torchtitan_deepseek>`
    * :doc:`Training a model with Primus <./notebooks/pretrain/training_with_primus>`
    * :doc:`SE(3)-Transformer overview <./notebooks/pretrain/se3transform_intro>`

  .. grid-item-card:: GPU development and optimization tutorials

    * :doc:`Quark MXFP4 quantization for vLLM <./notebooks/gpu_dev_optimize/mxfp4_quantization_quark_vllm>`
    * :doc:`GPU kernel development and assessment with Helion <./notebooks/gpu_dev_optimize/helion_gpu_kernel_dev>`
    * :doc:`MLA decoding kernel of AITER library <./notebooks/gpu_dev_optimize/aiter_mla_decode_kernel>`
    * :doc:`Kernel development and optimization with Triton <./notebooks/gpu_dev_optimize/triton_kernel_dev>`
    * :doc:`Profiling Llama-4 inference with vLLM <./notebooks/gpu_dev_optimize/llama4_profiling_vllm>`
    * :doc:`FP8 quantization with AMD Quark for vLLM <./notebooks/gpu_dev_optimize/fp8_quantization_quark_vllm>`

To contribute to the documentation, see
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find ROCm licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
