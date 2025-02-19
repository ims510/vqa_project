import pandas as pd
import ast
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm 

# Index(['image_id', 'split', 'filename', 'qa_id', 'question',
#        'multiple_choices', 'answer', 'type', 'feature_vector',
#        'question_embedding', 'choice_embeddings'],
#       dtype='object')

df = pd.read_pickle("data/df_with_features_and_embeddings.pkl")
# df = pd.read_csv("data/test.csv", encoding='UTF-8')
# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Process the DataFrame
new_rows = []
for index, row in tqdm(df.iterrows()):
    multiple_choices = row['multiple_choices']
    choice_embeddings = row['choice_embeddings']
    true_answer = row['answer']
    new_row = row.copy()
    new_row['multiple_choices'] = true_answer
    new_row['answer'] = 'TRUE'
    new_row['choice_embeddings'] = get_bert_embedding(true_answer)
    new_rows.append(new_row)
    for i, choice in enumerate(multiple_choices):
        new_row = row.copy()
        new_row['multiple_choices'] = choice
        new_row['answer'] = 'TRUE' if choice == row['answer'] else 'FALSE'
        new_row['choice_embeddings'] = choice_embeddings[i] 
        new_rows.append(new_row)
    

# Create the new DataFrame
new_df = pd.DataFrame(new_rows)

new_df.to_pickle("data/df_true_false_split.pkl")

