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

import json
import time
import requests
import datetime
from typing import Dict, Optional, Union

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step(enable_cache=False)
def openpipe_finetuning_starter(
    dataset_id: str,
    model_name: str,
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    openpipe_api_key: str = None,
    base_url: str = "https://api.openpipe.ai/api/v1",
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
) -> Dict:
    """Start a fine-tuning job on OpenPipe.

    This step initiates a fine-tuning job on OpenPipe and optionally waits for its completion.
    If waiting for completion, it will poll the API to get status updates and detailed model
    information including any error messages.

    Args:
        dataset_id: ID of the dataset to use for fine-tuning
        model_name: Name to assign to the fine-tuned model
        base_model: Base model to fine-tune
        openpipe_api_key: OpenPipe API key
        base_url: OpenPipe API base URL
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
    logger.info(f"Starting fine-tuning job for model: {model_name} with base model: {base_model}")
    
    # Set up headers for API requests
    headers = {
        "Authorization": f"Bearer {openpipe_api_key}",
        "Content-Type": "application/json"
    }
    
    # Check if model already exists and handle accordingly
    model_exists = False
    original_model_name = model_name
    
    try:
        # Try to fetch the model to see if it exists
        model_url = f"{base_url}/models/{model_name}"
        model_response = requests.get(model_url, headers=headers)
        
        if model_response.status_code == 200:
            model_exists = True
            
            if force_overwrite:
                # Delete the existing model
                logger.info(f"Model {model_name} already exists. Deleting it as force_overwrite=True")
                delete_response = requests.delete(model_url, headers=headers)
                
                if delete_response.status_code not in [200, 202, 204]:
                    logger.error(f"Failed to delete existing model {model_name}: {delete_response.text}")
                    raise Exception(f"Failed to delete existing model {model_name}")
                
                logger.info(f"Successfully deleted existing model {model_name}")
                model_exists = False  # Reset flag as model has been deleted
                
            elif auto_rename:
                # Generate a new unique name by appending a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{original_model_name}_{timestamp}"
                logger.info(f"Model {original_model_name} already exists. Using new name: {model_name}")
            else:
                # Fail with a more helpful error message
                logger.error(f"Model {model_name} already exists. Use auto_rename=True to generate a unique name "
                             f"or force_overwrite=True to replace the existing model.")
                raise Exception(f"Model {model_name} already exists. Use auto_rename=True or force_overwrite=True.")
    
    except requests.exceptions.RequestException as e:
        # 404 means model doesn't exist, which is good
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            model_exists = False
        else:
            logger.warning(f"Error checking if model exists: {str(e)}")
            # Continue anyway, we'll handle potential conflicts during creation
    
    # Prepare the fine-tuning configuration
    training_config = {
        "provider": "openpipe",
        "baseModel": base_model,
        "enable_sft": enable_sft,
        "enable_preference_tuning": enable_preference_tuning,
        "sft_hyperparameters": {
            "batch_size": batch_size,
            "learning_rate_multiplier": learning_rate_multiplier,
            "num_epochs": num_epochs,
        },
        "preference_hyperparameters": {}
    }
    
    # Add preference tuning specific parameters if enabled
    if enable_preference_tuning:
        training_config["preference_hyperparameters"] = {
            "variant": "DPO",
            "learning_rate_multiplier": learning_rate_multiplier,
            "num_epochs": num_epochs
        }
    
    # Create the fine-tuning job
    create_job_url = f"{base_url}/models"
    create_job_payload = {
        "datasetId": dataset_id,
        "slug": model_name,
        "pruningRuleIds": [],
        "trainingConfig": training_config,
        "defaultTemperature": default_temperature
    }
    
    try:
        # Start the fine-tuning job
        response = requests.post(
            create_job_url,
            json=create_job_payload,
            headers=headers
        )
        
        response.raise_for_status()
        job_data = response.json()
        
        job_id = job_data.get("id")
        if not job_id:
            raise ValueError("Failed to get job ID from response")
            
        logger.info(f"Successfully started fine-tuning job with ID: {job_id}")
        
        # Initialize result dict with basic information
        result = {
            "job_id": job_id,
            "model_name": model_name,
            "original_model_name": original_model_name if model_name != original_model_name else None,
            "renamed": model_name != original_model_name,
            "status": "pending",
            "model_info": None,
            "error_message": None
        }
        
        # Wait for the job to complete if requested
        if wait_for_completion:
            logger.info(f"Waiting for fine-tuning job {job_id} to complete...")
            
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            
            while time.time() - start_time < timeout_seconds:
                # Fetch detailed model information
                model_url = f"{base_url}/models/{model_name}"
                try:
                    model_response = requests.get(model_url, headers=headers)
                    model_response.raise_for_status()
                    model_info = model_response.json()
                    result["model_info"] = model_info
                    
                    # Extract status and error message if any
                    status = model_info.get("openpipe", {}).get("status", "UNKNOWN")
                    error_message = model_info.get("openpipe", {}).get("errorMessage")
                    
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
                        
                        # Log training parameters if available and it's the first time we're seeing them
                        hyperparams = model_info.get("openpipe", {}).get("hyperparameters", {})
                        if hyperparams and not result.get("logged_hyperparams"):
                            logger.info("Training parameters:")
                            for key, value in hyperparams.items():
                                logger.info(f"  {key}: {value}")
                            result["logged_hyperparams"] = True
                    
                    # If the job has completed (successfully or with error), break
                    if status in ["DEPLOYED", "ERROR"]:
                        break
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Failed to fetch model details, will retry: {str(e)}")
                    # Continue polling even if this request fails
                
                # Wait before checking again
                time.sleep(check_interval_seconds)
            
            # Final status check and log
            if result["status"] == "ready":
                logger.info(f"Fine-tuning job {job_id} completed successfully!")
            elif result["status"] == "failed":
                logger.error(f"Fine-tuning job {job_id} failed!")
                if result["error_message"]:
                    logger.error(f"Error message: {result['error_message']}")
            else:
                logger.warning(f"Fine-tuning job {job_id} is still running after timeout ({timeout_minutes} minutes)")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to start or monitor fine-tuning job: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        raise 

@step(enable_cache=False)
def openpipe_finetuning_starter_sdk(
    dataset_id: str,
    model_name: str,
    base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    openpipe_api_key: str = None,
    base_url: str = "https://api.openpipe.ai/api/v1",
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
) -> Dict:
    """Start a fine-tuning job on OpenPipe using the Python SDK.

    This step initiates a fine-tuning job on OpenPipe using the official Python SDK
    and optionally waits for its completion.

    Args:
        dataset_id: ID of the dataset to use for fine-tuning
        model_name: Name to assign to the fine-tuned model
        base_model: Base model to fine-tune
        openpipe_api_key: OpenPipe API key
        base_url: OpenPipe API base URL
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
    
    logger.info(f"Starting fine-tuning job for model: {model_name} with base model: {base_model}")
    
    # Initialize OpenPipe client
    op_client = OpenPipe(api_key=openpipe_api_key, base_url=base_url)
    
    # Initialize result dictionary
    result = {
        "model_name": model_name,
        "original_model_name": model_name,
        "renamed": False,
        "status": "pending",
        "model_info": None,
        "error_message": None
    }
    
    # Check if model already exists and handle accordingly
    original_model_name = model_name
    try:
        existing_model = op_client.get_model(model_slug=model_name)
        
        if existing_model:
            if force_overwrite:
                # Delete the existing model
                logger.info(f"Model {model_name} already exists. Deleting it as force_overwrite=True")
                op_client.delete_model(model_slug=model_name)
                logger.info(f"Successfully deleted existing model {model_name}")
            elif auto_rename:
                # Generate a new unique name by appending a timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{original_model_name}_{timestamp}"
                logger.info(f"Model {original_model_name} already exists. Using new name: {model_name}")
                result["renamed"] = True
                result["original_model_name"] = original_model_name
                result["model_name"] = model_name
            else:
                # Fail with a more helpful error message
                logger.error(f"Model {model_name} already exists. Use auto_rename=True to generate a unique name "
                             f"or force_overwrite=True to replace the existing model.")
                raise Exception(f"Model {model_name} already exists. Use auto_rename=True or force_overwrite=True.")
    except Exception as e:
        # If the exception is not about model existence, just log a warning and continue
        if not (hasattr(e, 'status_code') and e.status_code == 404):
            logger.warning(f"Error checking if model exists: {str(e)}")
    
    try:
        # Prepare training configuration
        training_config = {
            "provider": "openpipe",
            "baseModel": base_model,
            "enable_sft": enable_sft,
            "enable_preference_tuning": enable_preference_tuning,
            "sft_hyperparameters": {
                "batch_size": batch_size,
                "num_epochs": 4,
                "learning_rate_multiplier": learning_rate_multiplier,
            },
        }
        
        # Add preference tuning specific parameters if enabled
        if enable_preference_tuning:
            training_config["preference_hyperparameters"] = {
                "variant": "DPO",
                "learning_rate_multiplier": learning_rate_multiplier,
                "num_epochs": num_epochs
            }
        
        # Create the model
        model = op_client.create_model(
            dataset_id=dataset_id,
            slug=model_name,
            training_config=training_config,
            default_temperature=default_temperature
        )
        
            
        logger.info(f"Successfully started fine-tuning job with ID: {model.id}")
        result["job_id"] = model.id
        
        # Wait for the job to complete if requested
        if wait_for_completion:
            logger.info(f"Waiting for fine-tuning job {model.id} to complete...")
            
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            
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
                    
                    # If the job has completed (successfully or with error), break
                    if status in ["DEPLOYED", "ERROR"]:
                        break
                
                except Exception as e:
                    logger.warning(f"Failed to fetch model details, will retry: {str(e)}")
                
                # Wait before checking again
                time.sleep(check_interval_seconds)
            
            # Final status check and log
            if result["status"] == "ready":
                logger.info(f"Fine-tuning job {model.id} completed successfully!")
            elif result["status"] == "failed":
                logger.error(f"Fine-tuning job {model.id} failed!")
                if result["error_message"]:
                    logger.error(f"Error message: {result['error_message']}")
            else:
                logger.warning(f"Fine-tuning job {model.id} is still running after timeout ({timeout_minutes} minutes)")
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to start or monitor fine-tuning job: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        raise 
