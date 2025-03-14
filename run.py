# Apache Software License 2.0
# 
# Copyright (c) ZenML GmbH 2025. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import os
import json
import requests
from typing import Optional, List

import click
from zenml.client import Client
from zenml.logger import get_logger

from pipelines import openpipe_finetuning

logger = get_logger(__name__)


@click.command(
    help="""
ZenML OpenPipe Fine-Tuning Pipeline.

Run the ZenML OpenPipe fine-tuning pipeline.

Examples:

  \b
  # Run the OpenPipe fine-tuning pipeline with the toy dataset
    python run.py --openpipe-api-key=opk-your-api-key
  
  \b
  # Run with a specific model name and base model
    python run.py --openpipe-api-key=opk-your-api-key --model-name=my-model --base-model=meta-llama/Meta-Llama-3-70B-Instruct
  
  \b
  # Use your own CSV dataset
    python run.py --openpipe-api-key=opk-your-api-key --data-source=path/to/data.csv
  
  \b
  # Get detailed information about a model (useful for diagnosing failures)
    python run.py --openpipe-api-key=opk-your-api-key --model-name=my-model --fetch-details-only
    
  \b
  # Force overwrite an existing model with the same name
    python run.py --openpipe-api-key=opk-your-api-key --model-name=my-model --force-overwrite

"""
)
@click.option(
    "--openpipe-api-key",
    default=None,
    type=click.STRING,
    help="The OpenPipe API key. If not provided, it will try to use the OPENPIPE_API_KEY environment variable.",
)
@click.option(
    "--dataset-name",
    default="ultra_customer_service",
    type=click.STRING,
    help="Name for the OpenPipe dataset.",
)
@click.option(
    "--model-name",
    default="customer_service_assistant",
    type=click.STRING,
    help="Name for the fine-tuned model.",
)
@click.option(
    "--base-model",
    default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    type=click.STRING,
    help="Base model to fine-tune.",
)
@click.option(
    "--system-prompt",
    default="You are a helpful customer service assistant for Ultra electronics products.",
    type=click.STRING,
    help="System prompt to use for all examples.",
)
@click.option(
    "--data-source",
    default="toy",
    type=click.STRING,
    help="Source of the data - 'toy' for built-in example or path to a CSV file.",
)
@click.option(
    "--sample-size",
    default=30,
    type=click.INT,
    help="Number of samples to generate for the toy dataset.",
)
@click.option(
    "--metadata-columns",
    default=["product"],
    type=click.STRING,
    multiple=True,
    help="Optional columns to include as metadata (can be specified multiple times).",
)
@click.option(
    "--wait-for-completion/--no-wait-for-completion",
    default=True,
    help="Whether to wait for the fine-tuning job to complete.",
)
@click.option(
    "--verbose-logs/--no-verbose-logs",
    default=True,
    help="Whether to log detailed model information during polling.",
)
@click.option(
    "--auto-rename/--no-auto-rename",
    default=True,
    help="Whether to automatically append a timestamp to model name if it already exists.",
)
@click.option(
    "--force-overwrite",
    is_flag=True,
    default=False,
    help="Delete existing model with the same name before creating new one.",
)
@click.option(
    "--fetch-details-only",
    is_flag=True,
    default=False,
    help="Only fetch model details without running the fine-tuning pipeline.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--use-sdk",
    is_flag=True,
    default=False,
    help="Use the Python OpenPipe SDK instead of direct API calls.",
)
def main(
    openpipe_api_key: Optional[str] = None,
    dataset_name: str = "ultra_customer_service",
    model_name: str = "customer_service_assistant",
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    system_prompt: str = "You are a helpful customer service assistant for Ultra electronics products.",
    data_source: str = "toy",
    sample_size: int = 30,
    metadata_columns: List[str] = ["product"],
    wait_for_completion: bool = True,
    verbose_logs: bool = True,
    auto_rename: bool = True,
    force_overwrite: bool = False,
    fetch_details_only: bool = False,
    no_cache: bool = False,
    use_sdk: bool = False,
):
    """Main entry point for the OpenPipe fine-tuning pipeline.

    This entrypoint runs the OpenPipe fine-tuning pipeline with the specified parameters.

    Args:
        openpipe_api_key: The OpenPipe API key.
        dataset_name: Name for the OpenPipe dataset.
        model_name: Name for the fine-tuned model.
        base_model: Base model to fine-tune.
        system_prompt: System prompt to use for all examples.
        data_source: Source of the data - "toy" for built-in example or path to a CSV file.
        sample_size: Number of samples to generate for the toy dataset.
        metadata_columns: Optional columns to include as metadata.
        wait_for_completion: Whether to wait for the fine-tuning job to complete.
        verbose_logs: Whether to log detailed model information during polling.
        auto_rename: If True, automatically append a timestamp to model name if it already exists.
        force_overwrite: If True, delete existing model with the same name before creating new one.
        fetch_details_only: Only fetch model details without running the fine-tuning pipeline.
        no_cache: If `True` cache will be disabled.
        use_sdk: If `True` use the Python OpenPipe SDK instead of direct API calls.
    """
    client = Client()

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )

    # Get OpenPipe API key from environment if not provided
    if not openpipe_api_key:
        openpipe_api_key = os.environ.get("OPENPIPE_API_KEY")
        if not openpipe_api_key:
            logger.error("OpenPipe API key not provided. Please set --openpipe-api-key "
                         "or the OPENPIPE_API_KEY environment variable.")
            return
    
    # Check for conflicting options
    if force_overwrite and auto_rename:
        logger.warning("Both force_overwrite and auto_rename are enabled. force_overwrite will take precedence.")
    
    # If fetch_details_only is True, just fetch model details without running the pipeline
    if fetch_details_only:
        logger.info(f"Fetching details for model: {model_name}")
        
        # Set up headers for API request
        headers = {
            "Authorization": f"Bearer {openpipe_api_key}",
            "Content-Type": "application/json"
        }
        
        # Construct the URL
        base_url = "https://api.openpipe.ai/api/v1"
        url = f"{base_url}/models/{model_name}"
        
        try:
            # Make the API request
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            model_info = response.json()
            
            # Log important model information
            status = model_info.get("openpipe", {}).get("status", "UNKNOWN")
            error_message = model_info.get("openpipe", {}).get("errorMessage")
            base_model = model_info.get("openpipe", {}).get("baseModel", "unknown")
            created = model_info.get("created", "unknown")
            
            logger.info(f"Model: {model_name}")
            logger.info(f"Status: {status}")
            logger.info(f"Base model: {base_model}")
            logger.info(f"Created: {created}")
            
            if status == "ERROR" and error_message:
                logger.error(f"Error message: {error_message}")
            
            # Log training parameters if available
            hyperparams = model_info.get("openpipe", {}).get("hyperparameters", {})
            if hyperparams:
                logger.info("Training parameters:")
                for key, value in hyperparams.items():
                    logger.info(f"  {key}: {value}")
            
            # Print full JSON response for detailed debugging
            logger.info(f"Full model details: {json.dumps(model_info, indent=2)}")
            return
        except Exception as e:
            logger.error(f"Failed to fetch model details: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status code: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            return

    # Execute OpenPipe Fine-tuning Pipeline
    pipeline_args = {}
    if no_cache:
        pipeline_args["enable_cache"] = False
    pipeline_args["config_path"] = os.path.join(
        config_folder, "openpipe_finetuning.yaml"
    )
    
    # Set up run arguments
    run_args_openpipe = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "base_model": base_model,
        "system_prompt": system_prompt,
        "data_source": data_source,
        "sample_size": sample_size,
        "metadata_columns": list(metadata_columns),
        "wait_for_completion": wait_for_completion,
        "verbose_logs": verbose_logs,
        "auto_rename": auto_rename,
        "force_overwrite": force_overwrite,
        "openpipe_api_key": openpipe_api_key,
        "use_sdk": use_sdk,
    }
    
    # Run the pipeline
    openpipe_finetuning.with_options(**pipeline_args)(**run_args_openpipe)
    


if __name__ == "__main__":
    main()
