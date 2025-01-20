.. meta::
   :description: The AI Developer Hub provides tutorials and guides for training, fine-tuning, and inference
   :keywords: AI, ROCm, developers, tutorials, guides, training, fine-tuning, inference

.. _index:

****************************************
AI Developer Hub documents and tutorials
****************************************

The AI Developer Hub provides tutorials and guides for training, fine-tuning, and inference.
It leverages popular ML frameworks on AMD GPUs.

The AI Developer Hub public repository is located at `<https://github.com/ROCm/gpuaidev-docs>`_.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Inference tutorials

    * :doc:`Running inference with Hugging Face Transformers <./notebooks/inference/1_inference_ver3_HF_transformers>`
    * :doc:`Running Hugging Face text generation inference with Llama 3.1 8B <./notebooks/inference/2_inference_ver3_HF_TGI>`
    * :doc:`Deploying Llama 3.1 8B using vLLM <./notebooks/inference/3_inference_ver3_HF_vllm>`
    * :doc:`Building a chatbot and rap bot with vLLM <./notebooks/inference/rapbot_vllm>`
    * :doc:`Constructing a RAG system using LlamaIndex and Ollama <./notebooks/inference/rag_ollama_llamaindex>`

  .. grid-item-card:: Fine-tuning tutorials

    * :doc:`Fine-tuning with the Hugging Face ecosystem <./notebooks/fine_tune/fine_tuning_lora_qwen2vl>`
    * :doc:`Fine-Tuning Llama-3.2 3B with LoRA <./notebooks/fine_tune/LoRA_Llama-3.2>`
    * :doc:`Fine-Tuning Llama-3.1 with QLoRA <./notebooks/fine_tune/QLoRA_Llama-3.1>`

  .. grid-item-card:: Pretraining tutorials

    * :doc:`Pretrain an OLMo model with PyTorch FSDP <./notebooks/pretrain/torch_fsdp>`
    * :doc:`Pretraining with Megatron-LM <./notebooks/pretrain/setup_tutorial>`
    * :doc:`Training Llama 3.1 8B with Megatron-LM <./notebooks/pretrain/train_llama_mock_data>`

To contribute to the documentation, see
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
