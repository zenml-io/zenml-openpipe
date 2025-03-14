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
from typing import List, Optional

import pandas as pd
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def openpipe_data_converter(
    data: pd.DataFrame,
    system_prompt: str = "You are a helpful assistant",
    user_column: str = "question",
    assistant_column: str = "answer",
    split_ratio: float = 0.9,
    metadata_columns: Optional[List[str]] = None,
    output_path: str = "openpipe_data.jsonl",
) -> str:
    """Convert data to OpenPipe JSONL format.

    This step converts a pandas DataFrame to the OpenPipe JSONL format
    for fine-tuning.

    Args:
        data: Input DataFrame containing the training data
        system_prompt: The system prompt to use for all examples
        user_column: The column name containing the user messages
        assistant_column: The column name containing the assistant responses
        split_ratio: Ratio of train/test split (e.g., 0.9 means 90% train, 10% test)
        metadata_columns: Optional list of columns to include as metadata
        output_path: Path to save the JSONL file

    Returns:
        The path to the generated JSONL file
    """
    logger.info(f"Converting data to OpenPipe JSONL format and saving to {output_path}")

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Prepare the JSONL data
    jsonl_data = []

    for i, row in data.iterrows():
        # Determine the split (TRAIN or TEST)
        split = "TRAIN" if i < len(data) * split_ratio else "TEST"

        # Create the messages format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(row[user_column])},
            {"role": "assistant", "content": str(row[assistant_column])},
        ]

        # Create metadata if specified
        metadata = {}
        if metadata_columns:
            for col in metadata_columns:
                if col in row:
                    metadata[col] = str(row[col])

        # Create the entry
        entry = {
            "messages": messages,
            "split": split,
        }

        # Add metadata if present
        if metadata:
            entry["metadata"] = metadata

        jsonl_data.append(entry)

    # Write the JSONL file
    with open(output_path, "w") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        f"Successfully created {len(jsonl_data)} examples "
        f"({sum(1 for x in jsonl_data if x['split'] == 'TRAIN')} train, "
        f"{sum(1 for x in jsonl_data if x['split'] == 'TEST')} test)"
    )

    return output_path
