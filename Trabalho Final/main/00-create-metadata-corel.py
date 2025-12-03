#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create metadata.jsonl for Corel dataset
"""

import json
from pathlib import Path

# Class descriptions
CLASS_PROMPTS = {
    "0001": "british royal guards in red uniforms, ceremonial military parade",
    "0002": "steam locomotive train, vintage railway engine",
    "0003": "desserts and sweets, pastries and confectionery",
    "0004": "fresh salad with vegetables and fruits, healthy food",
    "0005": "winter snow scene, snowy landscape",
    "0006": "sunset over water, orange sky at dusk"
}

def create_metadata(data_dir="./data/corel", output_file="./data/corel/metadata.jsonl"):
    data_dir = Path(data_dir)
    image_files = sorted(list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg")))
    
    print(f"Found {len(image_files)} images")
    
    metadata = []
    for img_path in image_files:
        class_id = img_path.stem.split("_")[0]
        prompt = CLASS_PROMPTS.get(class_id, "corel image")
        
        metadata.append({
            "file_name": img_path.name,
            "text": prompt
        })
    
    # Write JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    
    print(f"Created metadata: {output_path}")

if __name__ == "__main__":
    create_metadata()