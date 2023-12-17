import os
import pandas as pd
import pandas as pd
import torch
import pinecone
import ast
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Set the API key
os.environ['OPENAI_API_KEY'] = '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'
os.environ['PINECONE_API_KEY'] = 'e20aa772-b33e-478e-a86d-31e5eb80d9dc'

OPENAI_API_KEY = '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'
PINECONE_API_KEY = 'e20aa772-b33e-478e-a86d-31e5eb80d9dc'

def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def string_to_list(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string  # Return the original string if it's not a list representation
    
df = pd.read_csv('enhanced_london_restaurants_tripadvisor.csv')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

batch_size = 100  # or another size that fits your memory constraints
all_embeddings = []

# Apply the conversion to the 'original_location' column
df['original_location'] = df['original_location'].apply(string_to_list).apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['combined_text'] = df.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in df.columns if col not in ['embeddings', 'combined_text']]), axis=1)

for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['combined_text'][i:i+batch_size].tolist()  # Convert DataFrame slice to a list of texts
    embeddings = generate_embeddings(batch_texts)  # Generate embeddings for the batch
    all_embeddings.extend(embeddings)

df['embeddings'] = all_embeddings

pinecone.init(api_key = PINECONE_API_KEY, environment="gcp-starter")
index_name = "tripadvisor-index"

embedding_dimension = len(df['embeddings'].iloc[6000])
print(f"The dimensionality of the embeddings is: {embedding_dimension}")

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=embedding_dimension)

index = pinecone.Index(index_name)

data_to_upload = []

# Iterate through each row in the DataFrame to prepare the data
for i, row in df.iterrows():
    metadata = str(row["combined_text"]) 
    vector = row['embeddings'].tolist()
    data_to_upload.append((str(i), vector, metadata))

# Upload data in batches to the Pinecone index
batch_size = 100
for i in range(0, len(data_to_upload), batch_size):
    batch = [(id, vec, {"metadata": metadata}) for id, vec, metadata in data_to_upload[i:i+batch_size]]
    index.upsert(vectors=batch)

print("########## DATA LOADING COMPLETE ##########")