<div align="center">
  <h1 align="center">ZenML ❤️ OpenPipe: Fine-Tune LLMs with MLOps Best Practices</h1>
  <h3 align="center">ZenML + OpenPipe brings production-grade MLOps to your LLM fine-tuning workflows</h3>
</div>

<div align="center">
  <br />
  <p>
    <img src="https://ai-infrastructure.org/wp-content/uploads/2022/08/ZenML-Logo.png" alt="ZenML Logo" height="100">
    &nbsp;&nbsp;&nbsp;&nbsp;
    <img src="https://bookface-images.s3.amazonaws.com/logos/fb9d92257bfc16c8162d539e84b1a125614976f3.png?1693238562" alt="OpenPipe Logo" height="100">
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

## 🗂️ Bringing Your Own Data

The integration supports using your own custom datasets for fine-tuning. Here's how to prepare and use your data:

### Data Format Requirements

Your CSV file should include at minimum these two columns:
- A column with user messages/questions (default: `question`)
- A column with assistant responses/answers (default: `answer`)

Example CSV structure:
```csv
question,answer,product
"How do I turn on my Ultra TV?","Press the power button on the remote or on the bottom right of the TV.",television
"Is my Ultra SmartWatch waterproof?","Yes, the Ultra SmartWatch is water-resistant up to 50 meters.",smartwatch
```

### Understanding the Data Transformation Process

When you provide your CSV file, the pipeline automatically:

1. **Reads** your CSV data
2. **Applies** the system prompt to all examples
3. **Converts** the data to OpenPipe's required JSONL format
4. **Splits** the data into training and testing sets

The final JSONL format looks like this (from the generated `openpipe_data.jsonl`):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful customer service assistant for Ultra electronics products."},
    {"role": "user", "content": "What is the price of the UltraPhone X?"},
    {"role": "assistant", "content": "The UltraPhone X is available for $999. Would you like to know about our financing options?"}
  ],
  "split": "TRAIN",
  "metadata": {"product": "UltraPhone X"}
}
```

### Step-by-Step Guide to Using Your Data

1. **Prepare your CSV file** with at least these columns:
   - A question/user message column (named `question` by default)
   - An answer/assistant response column (named `answer` by default)
   - Any additional metadata columns you want to include (optional)

2. **Run the pipeline** with your data file:
   ```bash
   python run.py --data-source=path/to/your/data.csv
   ```

3. **Check the results** in the ZenML dashboard or logs

Here's a complete example with all possible customizations:

```bash
python run.py \
  --data-source=my_customer_support_data.csv \
  --user-column=customer_query \
  --assistant-column=agent_response \
  --system-prompt="You are a helpful customer service assistant for Acme Corp." \
  --metadata-columns=product_category \
  --metadata-columns=customer_segment \
  --split-ratio=0.85
```

### Customizing Column Names

If your CSV uses different column names than the defaults, specify them with command-line arguments:

```bash
python run.py \
  --data-source=path/to/your/data.csv \
  --user-column=prompt \
  --assistant-column=completion
```

For example, if your CSV looks like this:
```csv
prompt,completion,category
"What's your return policy?","We offer a 30-day no-questions-asked return policy.",returns
"Do you ship internationally?","Yes, we ship to over 50 countries worldwide.",shipping
```

### Adding Metadata

You can include additional metadata columns in your CSV to enhance fine-tuning:

1. Add the columns to your CSV
2. Specify them when running the pipeline:
   ```bash
   python run.py --data-source=path/to/your/data.csv --metadata-columns=category --metadata-columns=difficulty
   ```

Metadata can help OpenPipe better understand the context of your training examples and can be useful for:
- Filtering and analyzing results
- Creating specialized versions of your model
- Understanding performance across different data categories

### Data Splitting

By default, the pipeline splits your data into training and evaluation sets using a 90/10 split. You can adjust this:

```bash
python run.py --data-source=path/to/your/data.csv --split-ratio=0.8
```

### System Prompt

You can set a custom system prompt that will be applied to all examples:

```bash
python run.py --data-source=path/to/your/data.csv --system-prompt="You are a customer service assistant for Ultra products."
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
