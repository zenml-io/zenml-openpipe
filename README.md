# OpenPipe ZenML Integration

This project integrates OpenPipe's fine-tuning capabilities with ZenML pipelines, allowing for streamlined development and deployment of fine-tuned language models.

## Overview

OpenPipe is a platform that allows for easy fine-tuning of large language models (LLMs) on your custom data. This integration with ZenML provides a structured pipeline approach to:

1. Convert your data to OpenPipe-compatible format
2. Create a dataset in OpenPipe
3. Upload your data to OpenPipe
4. Initiate fine-tuning on your chosen base model
5. Monitor the fine-tuning process

## Prerequisites

- Python 3.8+
- ZenML installed and configured
- An OpenPipe API key

## Installation

1. Clone this repository
2. Install the requirements:

```bash
pip install -r requirements.txt
```

3. Set your OpenPipe API key as an environment variable (optional):

```bash
export OPENPIPE_API_KEY=opk-your-api-key
```

## Usage

Run the pipeline with:

```bash
python run.py --openpipe-api-key=opk-your-api-key
```

### Command Line Options

- `--openpipe-api-key`: Your OpenPipe API key (required if not set as environment variable)
- `--dataset-name`: Name for the OpenPipe dataset (default: "zenml_dataset")
- `--model-name`: Name for the fine-tuned model (default: "zenml_finetuned_model")
- `--base-model`: Base model to fine-tune (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--system-prompt`: System prompt to use for all examples (default: "You are a helpful assistant")
- `--wait-for-completion/--no-wait-for-completion`: Whether to wait for the fine-tuning job to complete (default: wait)
- `--no-cache`: Disable caching for the pipeline run

## Customizing Data

The pipeline uses a data loader that loads data into a pandas DataFrame. The default implementation expects columns for 'question' and 'answer'. You may need to modify the data loader to match your data format.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
