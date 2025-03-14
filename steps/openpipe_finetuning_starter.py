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

import datetime
import time
from typing import Dict, Annotated

from zenml import step, log_metadata
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def openpipe_finetuning_starter(
    dataset_id: str,
    model_name: str,
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    openpipe_api_key: str = None,
    enable_sft: bool = True,
    enable_preference_tuning: bool = False,
    learning_rate_multiplier: float = 1.0,
    num_epochs: int = 10,
    batch_size: str = "auto",
    default_temperature: float = 0,
    wait_for_completion: bool = True,
    timeout_minutes: int = 120,
    check_interval_seconds: int = 10,
    verbose_logs: bool = True,
    auto_rename: bool = True,
    force_overwrite: bool = False,
) -> Annotated[Dict, "fine_tuning_result"]:
    """Start a fine-tuning job on OpenPipe using the Python SDK.

    This step initiates a fine-tuning job on OpenPipe using the official Python SDK
    and optionally waits for its completion.

    Args:
        dataset_id: ID of the dataset to use for fine-tuning
        model_name: Name to assign to the fine-tuned model
        base_model: Base model to fine-tune
        openpipe_api_key: OpenPipe API key
        enable_sft: Whether to enable supervised fine-tuning
        enable_preference_tuning: Whether to enable preference tuning
        learning_rate_multiplier: Learning rate multiplier for fine-tuning
        num_epochs: Number of epochs for fine-tuning
        batch_size: Batch size for fine-tuning ("auto" or an integer)
        default_temperature: Default temperature for the model
        wait_for_completion: Whether to wait for the fine-tuning job to complete
        timeout_minutes: Maximum time to wait for completion (in minutes)
        check_interval_seconds: Interval between status checks (in seconds)
        verbose_logs: Whether to log detailed model information during polling
        auto_rename: If True, automatically append a timestamp to model name if it already exists
        force_overwrite: If True, delete existing model with the same name before creating new one

    Returns:
        A dictionary with details about the fine-tuning job and model information
    """
    from openpipe.client import OpenPipe

    logger.info(
        f"Starting fine-tuning job for model: {model_name} with base model: {base_model}"
    )
    
    # Log fine-tuning configuration parameters
    log_metadata(
        metadata={
            "fine_tuning_config": {
                "dataset_id": dataset_id,
                "model_name": model_name,
                "base_model": base_model,
                "enable_sft": enable_sft,
                "enable_preference_tuning": enable_preference_tuning,
                "learning_rate_multiplier": learning_rate_multiplier,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "default_temperature": default_temperature,
                "wait_timeout_minutes": timeout_minutes if wait_for_completion else "N/A",
            },
            "model_management": {
                "auto_rename": auto_rename,
                "force_overwrite": force_overwrite,
            }
        }
    )

    # Initialize OpenPipe client
    op_client = OpenPipe(api_key=openpipe_api_key)

    # Initialize result dictionary
    result = {
        "model_name": model_name,
        "original_model_name": model_name,
        "renamed": False,
        "status": "pending",
        "model_info": None,
        "error_message": None,
    }

    # Check if model already exists and handle accordingly
    original_model_name = model_name
    try:
        existing_model = op_client.get_model(model_slug=model_name)

        if existing_model:
            if force_overwrite:
                # Delete the existing model
                logger.info(
                    f"Model {model_name} already exists. Deleting it as force_overwrite=True"
                )
                op_client.delete_model(model_slug=model_name)
                logger.info(f"Successfully deleted existing model {model_name}")
                log_metadata(
                    metadata={"model_action": "deleted_existing"}
                )
            elif auto_rename:
                # Generate a new unique name by appending a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{original_model_name}_{timestamp}"
                logger.info(
                    f"Model {original_model_name} already exists. Using new name: {model_name}"
                )
                result["renamed"] = True
                result["original_model_name"] = original_model_name
                result["model_name"] = model_name
                log_metadata(
                    metadata={
                        "model_rename": {
                            "action": "renamed",
                            "original_name": original_model_name,
                            "new_name": model_name
                        }
                    }
                )
            else:
                # Fail with a more helpful error message
                logger.error(
                    f"Model {model_name} already exists. Use auto_rename=True to generate a unique name "
                    f"or force_overwrite=True to replace the existing model."
                )
                log_metadata(
                    metadata={
                        "model_error": {
                            "action": "failed_existing_model",
                            "error": "Model already exists and neither auto_rename nor force_overwrite enabled"
                        }
                    }
                )
                raise Exception(
                    f"Model {model_name} already exists. Use auto_rename=True or force_overwrite=True."
                )
    except Exception as e:
        # If the exception is not about model existence, just log a warning and continue
        if not (hasattr(e, "status_code") and e.status_code == 404):
            logger.warning(f"Error checking if model exists: {str(e)}")
            log_metadata(
                metadata={"model_check_error": str(e)}
            )

    try:
        # Prepare training configuration
        training_config = {
            "provider": "openpipe",
            "baseModel": base_model,
            "enable_sft": enable_sft,
            "enable_preference_tuning": enable_preference_tuning,
            "sft_hyperparameters": {
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate_multiplier": learning_rate_multiplier,
            },
        }

        # Add preference tuning specific parameters if enabled
        if enable_preference_tuning:
            training_config["preference_hyperparameters"] = {
                "variant": "DPO",
                "learning_rate_multiplier": learning_rate_multiplier,
                "num_epochs": num_epochs,
            }

        # Create the model
        model = op_client.create_model(
            dataset_id=dataset_id,
            slug=model_name,
            training_config=training_config,
            default_temperature=default_temperature,
        )

        # Extract model ID
        job_id = model.id
        logger.info(f"Successfully started fine-tuning job with ID: {job_id}")
        result["job_id"] = job_id

        # Wait for the job to complete if requested
        if wait_for_completion:
            logger.info(f"Waiting for fine-tuning job {job_id} to complete...")
            
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            status_history = []

            while time.time() - start_time < timeout_seconds:
                try:
                    # Fetch the latest model information
                    model_info = op_client.get_model(model_slug=model_name)
                    result["model_info"] = model_info.dict()

                    # Extract status and error message if any
                    status = model_info.openpipe.status
                    error_message = model_info.openpipe.error_message
                    
                    # Map OpenPipe status to our result status
                    if status == "DEPLOYED":
                        result["status"] = "ready"
                    elif status == "ERROR" or error_message:
                        result["status"] = "failed"
                        result["error_message"] = error_message
                    else:
                        result["status"] = status.lower()

                    # Log detailed information if requested
                    if verbose_logs:
                        logger.info(f"Model: {model_name}, Status: {status}")
                        if error_message:
                            logger.error(f"Error message: {error_message}")

                        # Log training parameters if available
                        hyperparams = model_info.openpipe.hyperparameters
                        if hyperparams and not result.get("logged_hyperparams"):
                            logger.info("Training parameters:")
                            for key, value in hyperparams.items():
                                logger.info(f"  {key}: {value}")
                            result["logged_hyperparams"] = True
                            log_metadata(
                                metadata={"hyperparameters": hyperparams}
                            )

                    # If the job has completed (successfully or with error), break
                    if status in ["DEPLOYED", "ERROR"]:
                        break

                except Exception as e:
                    logger.warning(
                        f"Failed to fetch model details, will retry: {str(e)}"
                    )
                    log_metadata(
                        metadata={"polling_error": str(e)}
                    )

                # Wait before checking again
                time.sleep(check_interval_seconds)

            # Log status history
            log_metadata(
                metadata={
                    "completion_stats": {
                        "status_history": status_history,
                        "wait_duration_seconds": round(time.time() - start_time),
                        "completion_time": datetime.datetime.now().isoformat()
                    }
                }
            )

            # Final status check and log
            if result["status"] == "ready":
                logger.info(f"Fine-tuning job {job_id} completed successfully!")
                log_metadata(
                    metadata={"final_status": "success"}
                )
                
                # Log model info if available
                if result["model_info"]:
                    log_metadata(
                        metadata={
                            "model_deployment": {
                                "url": f"https://openpipe.ai/models/{model_name}",
                                "time": datetime.datetime.now().isoformat()
                            }
                        }
                    )
            elif result["status"] == "failed":
                logger.error(f"Fine-tuning job {job_id} failed!")
                if result["error_message"]:
                    logger.error(f"Error message: {result['error_message']}")
                    log_metadata(
                        metadata={
                            "job_failure": {
                                "status": "failed",
                                "error_message": result["error_message"]
                            }
                        }
                    )
            else:
                logger.warning(
                    f"Fine-tuning job {job_id} is still running after timeout ({timeout_minutes} minutes)"
                )
                log_metadata(
                    metadata={
                        "timeout": {
                            "status": "timed_out",
                            "minutes": timeout_minutes
                        }
                    }
                )

        return result

    except Exception as e:
        logger.error(f"Failed to start or monitor fine-tuning job: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
            
        log_metadata(
            metadata={
                "execution_error": {
                    "message": str(e),
                    "time": datetime.datetime.now().isoformat()
                }
            }
        )
        raise
