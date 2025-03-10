# Qwen3B-RAG: An Efficient RAG-based Knowledge Retrieval System

![GitHub tag](https://img.shields.io/github/tag/HasinduNimesh/Qwen3B-RAG.svg)
![Version](https://img.shields.io/badge/version-0.0.1-blue)
![License](https://img.shields.io/github/license/HasinduNimesh/Qwen3B-RAG--An-Efficient-RAG-based-Knowledge-Retrieval-System)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

## Overview
Qwen3B-RAG is a Retrieval-Augmented Generation (RAG) system built using the **Qwen-3B-Instruct** model. This project enables efficient information retrieval and summarization using **FAISS-based vector search** combined with **transformer-based summarization**.

The model processes large amounts of text (e.g., research papers, markdown documents, PDFs), generates structured embeddings, and dynamically retrieves the most relevant information upon query. This enhances the generation of concise, accurate, and contextually rich responses.

## Features 🚀
- **Retrieval-Augmented Generation (RAG)**: Uses **FAISS** to store and retrieve document embeddings dynamically.
- **Summarization & Question Answering**: Summarizes extracted chunks using **Google Pegasus-XSum**.
- **Hybrid Search with FAISS**: Efficient similarity search to find the most relevant document sections.
- **Easy Deployment**: Optimized for CPU and 4-bit quantized inference using **GGUF & llama-cpp-python**.
- **Training & Fine-Tuning**: Supports fine-tuning with **Unsloth** for custom knowledge adaptation.
- **Scalable**: Handles **large document processing** and can be integrated with **Gradio or Streamlit** for user interaction.

## Model Details 📚
**Base Model**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

**Fine-Tuned Model**: [HasinduNimesh/qwen3b-finetuned](https://huggingface.co/HasinduNimesh/qwen3b-finetuned)

### Training Configuration
- **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning (PEFT) using **Unsloth**.
- **Batch Size**: 1 (Gradient Accumulation = 4)
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 2048 tokens
- **Optimizer**: AdamW-8bit

### Model Outputs
- **Summarization**: Generates concise summaries from large text documents.
- **QA Responses**: Retrieves top-matching chunks and answers queries dynamically.

## Installation & Setup ⚡
### Step 1: Clone Repository
```bash
git clone https://github.com/HasinduNimesh/Qwen3B-RAG.git
cd Qwen3B-RAG
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the RAG System
```bash
python main.py
```

## Usage 🛠️
### Example Query
```python
from rag_pipeline import generate_answer

query = "Explain DeepSeek-R1's reinforcement learning process."
response = generate_answer(query)
print(response)
```

## Deployment 🌍
For a web-based UI, integrate with **Gradio**:
```bash
pip install gradio
```
```python
import gradio as gr

def chat_with_rag(query):
    return generate_answer(query)

demo = gr.Interface(fn=chat_with_rag, inputs="text", outputs="text")
demo.launch()
```

## Performance Benchmarks ⚡
| Model | Query Time (s) | Summarization Time (s) |
|--------|------------|------------------|
| Qwen3B | ~0.8s | ~2.5s |
| Qwen3B (Quantized) | ~0.4s | ~1.3s |

## Roadmap 🛤️
- [ ] Add support for **BM25 Hybrid Search** (ElasticSearch)
- [ ] Integrate **LangChain for multi-step reasoning**
- [ ] Improve **chunk overlap strategies** for better context retention
- [ ] Fine-tune with **domain-specific datasets**

## Contributing 🤝
We welcome contributions! Please check the [issues](https://github.com/HasinduNimesh/Qwen3B-RAG/issues) and submit PRs.

## License 📜
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙌
- [Alibaba Cloud - Qwen Model](https://huggingface.co/Qwen)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Unsloth Fine-Tuning](https://github.com/unslothai/unsloth)

## Model Usage 🚀
### Load Model in Python
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "HasinduNimesh/qwen3b-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

input_text = "Why is it necessary to filter out chain-of-thought outputs with mixed languages, long paragraphs, and code blocks?"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_length=256)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Use with Llama-CPP (4-bit GGUF)
```python
from llama_cpp import Llama

llm = Llama(model_path="unsloth.Q4_K_M.gguf", n_ctx=2048)
prompt = "Summarize the latest research on AI safety."
output = llm(prompt, max_tokens=200)
print(output["choices"][0]["text"])
```

## Future Improvements 🛠
- Improve dataset diversity: Add more diverse reasoning datasets
- Optimize retrieval: Enhance FAISS & BM25 hybrid retrieval
- Expand RL fine-tuning: Improve reward models for ORPO