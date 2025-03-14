<div align="center">
  <h1 align="center">ZenML ❤️ OpenPipe: Fine-Tune LLMs with MLOps Best Practices</h1>
  <h3 align="center">ZenML + OpenPipe brings production-grade MLOps to your LLM fine-tuning workflows</h3>
</div>

<div align="center">
  <br />
  <p>
    <img src="https://ai-infrastructure.org/wp-content/uploads/2022/08/ZenML-Logo.png" alt="ZenML Logo" height="50">
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="https://bookface-images.s3.amazonaws.com/logos/fb9d92257bfc16c8162d539e84b1a125614976f3.png?1693238562" alt="OpenPipe Logo" height="50">
  </p>
  <br />

  [![Python][python-shield]][python-url]
  [![ZenML][zenml-shield]][zenml-url]
  [![OpenPipe][openpipe-shield]][openpipe-url]
  [![Slack][slack-shield]][slack-url]

</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->


[license-url]: ./LICENSE
[python-shield]: https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue
[python-url]: https://www.python.org/downloads/
[zenml-shield]: https://img.shields.io/badge/ZenML-0.75.0%2B-431D93
[zenml-url]: https://github.com/zenml-io/zenml
[openpipe-shield]: https://img.shields.io/badge/OpenPipe-0.0.31%2B-orange
[openpipe-url]: https://www.openpipe.ai
[slack-shield]: https://img.shields.io/badge/-Slack-black.svg?logo=slack&colorB=7A3EF4
[slack-url]: https://zenml.io/slack-invite

---

## 🌟 What is This Repository?

This repository provides a powerful integration between [ZenML](https://zenml.io) and [OpenPipe](https://openpipe.ai), combining ZenML's production-grade MLOps orchestration with OpenPipe's specialized LLM fine-tuning capabilities.

**Perfect for teams who need to:**
- Create reproducible LLM fine-tuning pipelines
- Track all datasets, models, and experiments
- Deploy fine-tuned models to production with confidence
- Apply MLOps best practices to LLM workflows

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/zenml-io/zenml-openpipe.git
cd zenml-openpipe

# Install dependencies
pip install -r requirements.txt
```

### Set Up Your Environment

1. **OpenPipe Account**: [Sign up for OpenPipe](https://openpipe.ai/signup) to get your API key
2. **ZenML**: You can use ZenML in two ways:
   - **Open Source**: `pip install "zenml[server]"` and follow [self-hosting instructions](https://docs.zenml.io/getting-started/deploying-zenml)
   - **ZenML Pro** (optional): [Sign up](https://cloud.zenml.io/) for a managed experience with additional features

### Run Your First Pipeline

```bash
# Set your OpenPipe API key
export OPENPIPE_API_KEY=opk-your-api-key

# Run the pipeline with the toy dataset
python run.py
```

## ✨ Key Features

### 📊 End-to-End Fine-Tuning Pipeline

```python
@pipeline
def openpipe_finetuning(
    # Data parameters
    data_source: str = "toy",
    system_prompt: str = "You are a helpful assistant",
    
    # OpenPipe parameters
    model_name: str = "zenml_finetuned_model",
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    
    # Training parameters
    enable_sft: bool = True,
    num_epochs: int = 3,
    # ...and more
):
    # Load and prepare your data
    data = data_loader(...)
    jsonl_path = openpipe_data_converter(...)
    
    # Create OpenPipe dataset and start fine-tuning
    dataset_id = openpipe_dataset_creator(...)
    finetuning_result = openpipe_finetuning_starter(...)
    
    return finetuning_result
```

### 🔍 Complete Traceability

Every run of your fine-tuning pipeline tracks:
- Input data and processing
- Training configuration and hyperparameters
- Model performance and results

<div align="center">
  <img src="https://github.com/zenml-io/zenml/raw/main/docs/book/.gitbook/assets/finetune_zenml_home.png" width="600" alt="Pipeline Lineage">
</div>

### 🛠️ Flexible Customization

- Use toy datasets or bring your own data (CSV)
- Select from a variety of base models
- Customize supervised fine-tuning parameters
- Set up continuous training processes

## 📚 Advanced Usage

### Custom Data Source

```bash
# Use your own CSV dataset
python run.py --openpipe-api-key=opk-your-api-key --data-source=path/to/data.csv
```

### Model Selection

```bash
# Fine-tune Llama-3-70B instead of the default
python run.py --openpipe-api-key=opk-your-api-key --model-name=my-model --base-model=meta-llama/Meta-Llama-3-70B-Instruct
```

### Inspecting Model Details

```bash
# Get detailed information about an existing model
python run.py --openpipe-api-key=opk-your-api-key --model-name=my-model --fetch-details-only
```

## 🧩 How It Works

The integration leverages:

1. **ZenML's Pipeline Orchestration**: Handles workflow DAGs, artifact tracking, and reproducibility
2. **OpenPipe's LLM Fine-Tuning**: Provides state-of-the-art techniques for adapting foundation models


## 📖 Learn More

- [ZenML Documentation](https://docs.zenml.io/)
- [OpenPipe Documentation](https://docs.openpipe.ai/)
- [ZenML GitHub](https://github.com/zenml-io/zenml)

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md) for details on how to get started.

## 🆘 Getting Help

- Join the [ZenML Slack community](https://zenml.io/slack) 
- Ask questions in the **#help** channel or the **#openpipe** channel
- [Open an issue](https://github.com/zenml-io/zenml-openpipe/issues/new/choose) on GitHub

## 📜 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
