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

import random
from typing import Optional

import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def data_loader(
    random_state: int = 42,
    target: Optional[str] = None,
    sample_size: int = 30,
    data_source: str = "toy",
) -> Annotated[pd.DataFrame, "dataset"]:
    """Dataset loader step for OpenPipe fine-tuning.

    This step loads a dataset suitable for fine-tuning language models with OpenPipe.
    It can either load a toy example dataset of Q&A conversations or a custom dataset.

    Args:
        random_state: Random state for reproducibility
        target: Not used for this dataset, kept for compatibility
        sample_size: Number of samples to generate for the toy dataset
        data_source: Source of the data - "toy" for built-in example or path to a CSV file

    Returns:
        A DataFrame containing conversation data with question and answer columns
    """
    random.seed(random_state)

    if data_source == "toy":
        logger.info(f"Loading toy conversation dataset with {sample_size} samples")

        # Example knowledge base for a customer service assistant
        product_info = [
            {
                "name": "UltraPhone X",
                "price": "$999",
                "features": "5G, 6.7-inch display, 12MP camera, 256GB storage",
            },
            {
                "name": "UltraPhone SE",
                "price": "$699",
                "features": "4G, 5.8-inch display, 8MP camera, 128GB storage",
            },
            {
                "name": "UltraTablet Pro",
                "price": "$1299",
                "features": "10.5-inch display, 512GB storage, keyboard included",
            },
            {
                "name": "UltraWatch",
                "price": "$349",
                "features": "Health tracking, notifications, 2-day battery life",
            },
            {
                "name": "UltraBook Air",
                "price": "$1099",
                "features": "13-inch display, 8GB RAM, 256GB SSD, 10hr battery",
            },
        ]

        # Templates for questions
        question_templates = [
            "What is the price of the {product}?",
            "Tell me about the features of {product}",
            "How does the {product} compare to other models?",
            "Is the {product} compatible with my other devices?",
            "When will the {product} be back in stock?",
            "Do you offer any discounts on the {product}?",
            "What colors does the {product} come in?",
            "How long is the warranty for the {product}?",
            "Can I trade in my old device for a new {product}?",
            "What accessories are available for the {product}?",
        ]

        # Templates for answers
        answer_templates = {
            "What is the price of the {product}?": "The {product} is available for {price}. Would you like to know about our financing options?",
            "Tell me about the features of {product}": "The {product} offers {features}. Is there a specific feature you're interested in?",
            "How does the {product} compare to other models?": "The {product} ({price}) has {features}. Compared to other models, it offers a great balance of performance and value.",
            "Is the {product} compatible with my other devices?": "Yes, the {product} is compatible with all other Ultra devices and most third-party products. It supports standard connectivity options.",
            "When will the {product} be back in stock?": "We expect to receive more stock of the {product} within the next 2 weeks. Would you like to be notified when it's available?",
            "Do you offer any discounts on the {product}?": "We currently have a 10% discount for students and military personnel on the {product}, bringing the price from {price} to a reduced rate.",
            "What colors does the {product} come in?": "The {product} is available in Space Gray, Silver, and Midnight Blue colors. The Midnight Blue is our newest addition.",
            "How long is the warranty for the {product}?": "The {product} comes with a 1-year limited warranty. You can extend it to 3 years with our UltraCare+ protection plan.",
            "Can I trade in my old device for a new {product}?": "Absolutely! You can trade in your eligible device for credit toward your new {product}, potentially saving up to $300.",
            "What accessories are available for the {product}?": "We offer a range of accessories for the {product}, including cases, screen protectors, chargers, and adapters.",
        }

        # Generate conversations
        questions = []
        answers = []

        for _ in range(sample_size):
            product = random.choice(product_info)
            question_template = random.choice(question_templates)
            question = question_template.format(product=product["name"])

            # Find matching answer template or use a generic one
            if question_template in answer_templates:
                answer_template = answer_templates[question_template]
                answer = answer_template.format(
                    product=product["name"],
                    price=product["price"],
                    features=product["features"],
                )
            else:
                answer = f"Thank you for your question about the {product['name']}. A customer service representative will assist you shortly."

            questions.append(question)
            answers.append(answer)

        # Create DataFrame with conversation data
        data = pd.DataFrame(
            {
                "question": questions,
                "answer": answers,
                "product": [
                    product_info[i % len(product_info)]["name"]
                    for i in range(sample_size)
                ],
            }
        )

    else:
        # Load from a custom CSV file
        logger.info(f"Loading conversation dataset from {data_source}")
        try:
            data = pd.read_csv(data_source)
            required_columns = ["question", "answer"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"CSV file is missing required columns: {missing_columns}"
                )

        except Exception as e:
            logger.error(f"Failed to load data from {data_source}: {str(e)}")
            # Fallback to toy dataset
            logger.info(f"Falling back to toy dataset")
            return data_loader(
                random_state=random_state, sample_size=sample_size, data_source="toy"
            )

    logger.info(f"Loaded dataset with {len(data)} conversation examples")
    return data
