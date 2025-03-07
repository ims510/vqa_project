"""
This is the first script to run in the pipeline, as part of a VQA task, after downloading the dataset. (use bash scripts for this)
This script extracts data from the JSON file and processes the images and text to create feature vectors and embeddings.
The resulting DataFrame is saved as a pickle file for later use.
"""

import pandas as pd
import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import random
import numpy as np

# Constants - paths to data
DATASET_PATH = 'data/dataset.json'
IMAGES_PATH = 'data/images/'

def extract_data_from_json():
    """
    This function extracts data from the JSON file and creates a DataFrame.
    Each row in the DataFrame corresponds to a question-answer pair with associated metadata.

    For each question, the structure of the JSON file is as follows:
    "image_id": 1, 
          "question": "What is on the sidewalk's edge?", 
          "multiple_choices": [
            "Grass.", 
            "Fire hydrant.", 
            "Sign."
          ], 
          "qa_id": 68861, 
          "answer": "Lamp post and lamp.", 
          "type": "what"
    """
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
    """
    This function loads an image from the specified path, 
    applies the transformation and extracts the feature vector.
    Args:
        image_path (str): Path to the image file.
        model (nn.Module): Pre-trained model for feature extraction.
        transform (transforms.Compose): Transformations to apply to the image.
        device (torch.device): Device to run the model on.
    Returns:
        np.ndarray: Extracted feature vector.
    """
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0).to(device, dtype=torch.float32) # Adding batch dimension and move to device

    with torch.no_grad():
        features = model(img)
    
    return features.squeeze().cpu().numpy()  # Move back to CPU and convert to numpy

def add_feature_vectors_to_df(df, device):
    """
    This function adds feature vectors to the DataFrame by processing the images 
    using a pre-trained ResNet model.
    Args:
        df (pd.DataFrame): DataFrame containing the image filenames.
        device (torch.device): Device to run the model on.
    
    Returns:
        pd.DataFrame: DataFrame with added feature vectors.
    """
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
    """
    Embeds the input text using the specified tokenizer and model.
    
    Args:
        text (str): Text to embed.
        tokenizer (BertTokenizer): Tokenizer for the model.
        model (BertModel): Pre-trained BERT model.
        device (torch.device): Device to run the model on.
    
    Returns:
        np.ndarray: Text embedding.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move back to CPU and convert to numpy

def add_text_embeddings_to_df(df, device):
    """
    This function adds text embeddings to the DataFrame 
    by processing the text using a pre-trained BERT model.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    question_embeddings = []
    choice1_embeddings = []
    choice2_embeddings = []
    choice3_embeddings = []
    choice4_embeddings = []
    labels = []

    for _, row in tqdm(df.iterrows(), desc="Processing text", total=len(df)):
        question_embedding = embed_text(row['question'], tokenizer, model, device)
        question_embeddings.append(question_embedding)

        choices = row['multiple_choices']
        answer = row['answer']
        choice_embeddings = [
            embed_text(choices[0], tokenizer, model, device),
            embed_text(choices[1], tokenizer, model, device),
            embed_text(choices[2], tokenizer, model, device),
            embed_text(answer, tokenizer, model, device)
        ]

        # Shuffle the choices and keep track of the correct answer's new position
        indices = list(range(4))
        random.shuffle(indices)
        shuffled_embeddings = [choice_embeddings[i] for i in indices]
        correct_index = indices.index(3)

        choice1_embeddings.append(shuffled_embeddings[0])
        choice2_embeddings.append(shuffled_embeddings[1])
        choice3_embeddings.append(shuffled_embeddings[2])
        choice4_embeddings.append(shuffled_embeddings[3])

        # Create the label array
        label = [0, 0, 0, 0]
        label[correct_index] = 1
        labels.append(label)

    df['question_embedding'] = question_embeddings
    df['choice1_embedding'] = choice1_embeddings
    df['choice2_embedding'] = choice2_embeddings
    df['choice3_embedding'] = choice3_embeddings
    df['choice4_embedding'] = choice4_embeddings
    df['answer_array'] = labels
    return df

def flatten_and_convert_to_array(x):
    """
    This function flattens the input list or tuple and converts it to a numpy array.
    """
    if isinstance(x, (list, tuple)):
        # Convert to numpy array and squeeze
        squeezed = np.squeeze(np.array(x))
        return squeezed
    else:
        return x

def convert_answer_to_array(x):
    """
    This function converts the input list to a numpy array.
    """
    if isinstance(x, list):
        return np.array(x)
    else:
        return x

def inspect_and_clean_dataframe(df):
    """
    This function inspects the DataFrame and applies the necessary conversions.
    """
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame size: {df.size}")
    print("DataFrame columns and example data:")
    
    for column in df.columns:
        print(f"Inspecting column: {column}")
        if column.startswith("choice") and "embedding" in column or column in ["feature_vector", "question_embedding"]:
            print(f"Applying conversion to column: {column}")
            # Apply the conversion function
            df[column] = df[column].apply(flatten_and_convert_to_array)
            print(f"Converted {column} to numpy arrays")
        elif column == "answer":
            print(f"Converting {column} to numpy arrays")
            df[column] = df[column].apply(convert_answer_to_array)
            print(f"Converted {column} to numpy arrays")

        print(f"\nColumn: {column}")
        print(f"Data type: {df[column].dtype}")
        example_data = str(df[column].iloc[0])
        print(f"Example data: {example_data[:100]}")

    return df

# Check if GPU is available (my chip is m2pro)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Extract data from JSON file - each row of the data frame corresponds to a question
df = extract_data_from_json()
df_with_features = add_feature_vectors_to_df(df, device) # add image embeddings using resnet50
df_with_embeddings = add_text_embeddings_to_df(df_with_features, device) # add text embeddings using bert
clean_df = inspect_and_clean_dataframe(df_with_embeddings)
clean_df.to_pickle('data/final_clean_df.pkl')