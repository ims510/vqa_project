import pandas as pd
import json

# Constants - paths to data
DATASET_PATH = 'data/dataset.json'
IMAGES_PATH = 'data/images/'

def extract_data_from_json():
    with open(DATASET_PATH  , 'r') as file:
        data = json.load(file)
    
    records = []
    for image in data['images']:
        image_id = image['image_id']
        split = image['split']
        filename = image['filename']
        for qa_pair in image.get('qa_pairs', []):
            record = {
                'image_id': image_id,
                'split': split,
                'filename': filename,
                'qa_id': qa_pair['qa_id'],
                'question': qa_pair['question'],
                'multiple_choices': qa_pair['multiple_choices'],
                'answer': qa_pair['answer'],
                'type': qa_pair['type']
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    return df

df = extract_data_from_json()
print(df.head())