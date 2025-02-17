import pandas as pd
import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

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

def create_feature_vector(image_path, model, transform, device):
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(device, dtype=torch.float32)  # Add batch dimension, move to device, and ensure correct type

    with torch.no_grad():
        features = model(img)
    
    return features.squeeze().cpu().numpy()  # Move back to CPU and convert to numpy

def add_feature_vectors_to_df(df, device):
    # Load pre-trained ResNet model
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the last layer
    model.to(device)  # Move model to device
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
        feature_vector = create_feature_vector(image_path, model, transform, device)
        feature_vectors.append(feature_vector)
    
    df['feature_vector'] = feature_vectors
    return df

def embed_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move back to CPU and convert to numpy

def add_text_embeddings_to_df(df, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    question_embeddings = []
    choice_embeddings = []

    for _, row in tqdm(df.iterrows(), desc="Processing text", total=len(df)):
        question_embedding = embed_text(row['question'], tokenizer, model, device)
        question_embeddings.append(question_embedding)

        choices = row['multiple_choices']
        choice_embedding = [embed_text(choice, tokenizer, model, device) for choice in choices]
        choice_embeddings.append(choice_embedding)

    df['question_embedding'] = question_embeddings
    df['choice_embeddings'] = choice_embeddings
    return df

# Check if GPU is available (my chip is m2pro)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

df = extract_data_from_json()
df_with_features = add_feature_vectors_to_df(df, device)
df_with_embeddings = add_text_embeddings_to_df(df_with_features, device)
# df_with_embeddings = add_text_embeddings_to_df(df, device)
print(df_with_embeddings.head())
df_with_embeddings.to_pickle('data/df_with_features_and_embeddings.pkl')