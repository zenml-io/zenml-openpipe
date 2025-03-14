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

from typing import List, Optional

from zenml import pipeline
from zenml.logger import get_logger

from steps import (
    data_loader,
    openpipe_data_converter,
    openpipe_dataset_creator,
    openpipe_finetuning_starter,
)

logger = get_logger(__name__)


@pipeline
def openpipe_finetuning(
    # Data loading and preparation parameters
    random_state: int = 42,
    sample_size: int = 30,
    data_source: str = "toy",
    user_column: str = "question",
    assistant_column: str = "answer",
    system_prompt: str = "You are a helpful assistant",
    split_ratio: float = 0.9,
    metadata_columns: Optional[List[str]] = None,
    # OpenPipe dataset parameters
    dataset_name: str = "zenml_dataset",
    openpipe_api_key: Optional[str] = None,
    # Fine-tuning parameters
    model_name: str = "zenml_finetuned_model",
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    enable_sft: bool = True,
    enable_preference_tuning: bool = False,
    learning_rate_multiplier: float = 1.0,
    num_epochs: int = 3,
    batch_size: str = "auto",
    default_temperature: float = 0.7,
    wait_for_completion: bool = True,
    timeout_minutes: int = 120,
    verbose_logs: bool = True,
    auto_rename: bool = True,
    force_overwrite: bool = False,
):
    """
    OpenPipe fine-tuning pipeline.

    This pipeline loads data, converts it to OpenPipe format, creates an OpenPipe dataset,
    uploads the data, and initiates fine-tuning.

    Args:
        random_state: Random state for reproducibility
        sample_size: Number of samples to generate for the toy dataset
        data_source: Source of the data - "toy" for built-in example or path to a CSV file
        user_column: Column containing user messages
        assistant_column: Column containing assistant responses
        system_prompt: System prompt to use for all examples
        split_ratio: Train/test split ratio
        metadata_columns: Optional columns to include as metadata
        dataset_name: Name for the OpenPipe dataset
        openpipe_api_key: OpenPipe API key
        model_name: Name for the fine-tuned model
        base_model: Base model to fine-tune
        enable_sft: Whether to enable supervised fine-tuning
        enable_preference_tuning: Whether to enable preference tuning
        learning_rate_multiplier: Learning rate multiplier
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        default_temperature: Default temperature for the model
        wait_for_completion: Whether to wait for fine-tuning to complete
        timeout_minutes: Maximum time to wait for completion
        verbose_logs: Whether to log detailed model information during polling
        auto_rename: If True, automatically append a timestamp to model name if it already exists
        force_overwrite: If True, delete existing model with the same name before creating new one

    Returns:
        A dictionary with details about the fine-tuning job, including model information
    """
    # Load data
    data = data_loader(
        random_state=random_state,
        sample_size=sample_size,
        data_source=data_source,
    )

    # Convert data to OpenPipe format
    jsonl_path = openpipe_data_converter(
        data=data,
        system_prompt=system_prompt,
        user_column=user_column,
        assistant_column=assistant_column,
        split_ratio=split_ratio,
        metadata_columns=metadata_columns,
    )

    # Create OpenPipe dataset and upload data
    dataset_id = openpipe_dataset_creator(
        jsonl_path=jsonl_path,
        dataset_name=dataset_name,
        openpipe_api_key=openpipe_api_key,
    )

    # Start fine-tuning using the SDK implementation
    finetuning_result = openpipe_finetuning_starter(
        dataset_id=dataset_id,
        model_name=model_name,
        base_model=base_model,
        openpipe_api_key=openpipe_api_key,
        enable_sft=enable_sft,
        enable_preference_tuning=enable_preference_tuning,
        learning_rate_multiplier=learning_rate_multiplier,
        num_epochs=num_epochs,
        batch_size=batch_size,
        default_temperature=default_temperature,
        wait_for_completion=wait_for_completion,
        timeout_minutes=timeout_minutes,
        verbose_logs=verbose_logs,
        auto_rename=auto_rename,
        force_overwrite=force_overwrite,
    )

    return finetuning_result
