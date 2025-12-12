#!/usr/bin/env python
"""Quick script to check TruthfulQA dataset version and details."""

from datasets import load_dataset, get_dataset_config_info

print("Checking TruthfulQA dataset information...\n")

# Load the dataset the same way the app does
dataset = load_dataset("truthful_qa", "generation", split="validation")

print(f"Dataset size: {len(dataset)} questions")
print(f"\nDataset info:")
print(f"  Dataset name: {dataset.info.dataset_name if hasattr(dataset.info, 'dataset_name') else 'N/A'}")
print(f"  Version: {dataset.info.version if hasattr(dataset.info, 'version') else 'N/A'}")
print(f"  Description: {dataset.info.description[:200] if hasattr(dataset.info, 'description') else 'N/A'}...")

print(f"\nDataset features:")
for feature_name, feature_type in dataset.features.items():
    print(f"  - {feature_name}: {feature_type}")

print(f"\nFirst question sample:")
sample = dataset[0]
for key in sample.keys():
    value = sample[key]
    if isinstance(value, list):
        print(f"  {key}: {value[:2] if len(value) > 2 else value}... ({len(value)} items)")
    else:
        print(f"  {key}: {value}")

# Check if we can get config info
try:
    config_info = get_dataset_config_info("truthful_qa", "generation")
    print(f"\nConfig info:")
    print(f"  Config name: {config_info.config_name}")
    print(f"  Version: {config_info.version}")
except Exception as e:
    print(f"\nCould not get config info: {e}")
