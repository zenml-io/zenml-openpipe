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
import os
import requests
from typing import Dict, Optional

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def openpipe_dataset_creator(
    jsonl_path: str,
    dataset_name: str,
    openpipe_api_key: str,
    base_url: str = "https://api.openpipe.ai/api/v1",
) -> str:
    """Create an OpenPipe dataset and upload data.

    This step creates a new dataset in OpenPipe and uploads the JSONL data to it.

    Args:
        jsonl_path: Path to the JSONL file containing the data
        dataset_name: Name of the dataset to create
        openpipe_api_key: OpenPipe API key
        base_url: OpenPipe API base URL

    Returns:
        The ID of the created dataset
    """
    logger.info(f"Creating OpenPipe dataset: {dataset_name}")
    
    # Set up headers for API requests
    headers = {
        "Authorization": f"Bearer {openpipe_api_key}",
        "Content-Type": "application/json"
    }
    
    # Create the dataset
    create_dataset_url = f"{base_url}/datasets"
    create_dataset_payload = {"name": dataset_name}
    
    try:
        response = requests.post(
            create_dataset_url, 
            json=create_dataset_payload, 
            headers=headers
        )
        response.raise_for_status()
        dataset_id = response.json().get("id")
        
        if not dataset_id:
            raise ValueError("Failed to get dataset ID from response")
            
        logger.info(f"Successfully created dataset with ID: {dataset_id}")
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise
    
    # Upload the data
    try:
        # Read the JSONL file and prepare entries
        entries = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    entries.append(json.loads(line))
        
        # Upload data in batches to avoid potential API limitations
        batch_size = 100
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i+batch_size]
            
            upload_url = f"{base_url}/datasets/{dataset_id}/entries"
            upload_payload = {"entries": batch}
            
            response = requests.post(
                upload_url,
                json=upload_payload,
                headers=headers
            )
            response.raise_for_status()
            
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(entries)-1)//batch_size + 1} "
                        f"({len(batch)} entries)")
    
        logger.info(f"Successfully uploaded {len(entries)} entries to dataset {dataset_id}")
    except Exception as e:
        logger.error(f"Failed to upload data: {str(e)}")
        raise
    
    return dataset_id 