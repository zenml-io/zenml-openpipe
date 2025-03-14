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
from typing import Annotated

from openpipe.client import OpenPipe
from zenml import log_metadata, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def openpipe_dataset_creator(
    jsonl_path: str,
    dataset_name: str,
    openpipe_api_key: str,
) -> Annotated[str, "dataset_id"]:
    """Create an OpenPipe dataset and upload data.

    This step creates a new dataset in OpenPipe and uploads the JSONL data to it,
    using the OpenPipe Python SDK.

    Args:
        jsonl_path: Path to the JSONL file containing the data
        dataset_name: Name of the dataset to create
        openpipe_api_key: OpenPipe API key

    Returns:
        The ID of the created dataset
    """
    logger.info(f"Creating OpenPipe dataset: {dataset_name}")

    # Log dataset creation metadata
    log_metadata(
        metadata={
            "dataset_info": {
                "dataset_name": dataset_name,
                "input_file": jsonl_path,
                "api_endpoint": "https://api.openpipe.ai/api/v1",
            }
        }
    )

    # Initialize OpenPipe client
    op_client = OpenPipe(api_key=openpipe_api_key)

    try:
        # Create the dataset
        dataset = op_client.create_dataset(name=dataset_name)
        dataset_id = dataset.id

        if not dataset_id:
            raise ValueError("Failed to get dataset ID from response")

        logger.info(f"Successfully created dataset with ID: {dataset_id}")

        # Log the created dataset ID
        log_metadata(metadata={"openpipe_dataset_id": dataset_id})
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise

    # Upload the data
    try:
        # Read the JSONL file and prepare entries
        entries = []
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    entries.append(json.loads(line))

        # Upload data in batches to avoid potential API limitations
        batch_size = 100
        total_batches = (len(entries) - 1) // batch_size + 1

        for i in range(0, len(entries), batch_size):
            batch = entries[i : i + batch_size]
            batch_num = i // batch_size + 1

            op_client.create_dataset_entries(dataset_id=dataset_id, entries=batch)

            logger.info(
                f"Uploaded batch {batch_num}/{total_batches} ({len(batch)} entries)"
            )

        logger.info(
            f"Successfully uploaded {len(entries)} entries to dataset {dataset_id}"
        )

        # Log final stats
        log_metadata(
            metadata={
                "upload_results": {
                    "total_examples_uploaded": len(entries),
                    "upload_status": "complete",
                }
            }
        )
    except Exception as e:
        logger.error(f"Failed to upload data: {str(e)}")
        # Log error details
        log_metadata(
            metadata={"upload_error": {"status": "failed", "error_message": str(e)}}
        )
        raise

    return dataset_id
