.. meta::
   :description: The AI Developer Hub provides tutorials and guides for training, fine-tuning, and inference
   :keywords: AI, ROCm, developers, tutorials, guides, training, fine-tuning, inference

.. _index:

****************************************
Tutorials for AI developers
****************************************

The AI Developer Hub contains AMD ROCm tutorials in Jupyter Notebook format for training, fine-tuning, and inference.
It leverages popular machine learning frameworks on AMD GPUs.

These tutorials are organized into three main categories:

*  **Fine-tuning**: Examples and guides for fine-tuning machine learning models.
*  **Pretraining**: Tutorials on pretraining models from scratch.
*  **Inference**: Resources for running inference with trained models.

The AI Developer Hub public repository is located at `<https://github.com/ROCm/gpuaidev-docs>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Inference tutorials

    * :doc:`Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Hugging Face TGI with Llama-3.1 8B <./notebooks/inference/2_inference_ver3_HF_TGI>`
    * :doc:`Llama-3.1 8B using vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`Chatbot and rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`RAG system using LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`

  .. grid-item-card:: Fine-tuning tutorials

    * :doc:`Hugging Face ecosystem <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Llama-3.2 3B with LoRA <./notebooks/fine_tune/LoRA_Llama-3.2>`
    * :doc:`Llama-3.1 with QLoRA <./notebooks/fine_tune/QLoRA_Llama-3.1>`

  .. grid-item-card:: Pretraining tutorials

    * :doc:`OLMo model with PyTorch FSDP <./notebooks/pretrain/torch_fsdp>`
    * :doc:`Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`Llama-3.1 8B with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`

To contribute to the documentation, see
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.