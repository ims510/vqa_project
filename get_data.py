import pandas as pd
import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

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

def create_feature_vector(image_path, model, transform):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(img)
    
    return features.squeeze().numpy()

def add_feature_vectors_to_df(df):
    # Load pre-trained ResNet model
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) # got numbers from this post https://stackoverflow.com/questions/62117707/extract-features-from-pretrained-resnet50-in-pytorch

    feature_vectors = []

    for filename in tqdm(df['filename'], desc="Processing images"):
        image_path = os.path.join(IMAGES_PATH, filename)
        feature_vector = create_feature_vector(image_path, model, transform)
        feature_vectors.append(feature_vector)
    
    df['feature_vector'] = feature_vectors
    return df

df = extract_data_from_json()
print(df.head())
df_with_features = add_feature_vectors_to_df(df)
print(df_with_features.head())
df_with_features.to_pickle('data/df_with_features.pkl')