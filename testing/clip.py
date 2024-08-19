# Get a segmentation fault error when this runs.
# Can't figure out why. Will come back and play with it later.

import faiss
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("Loading model and processor...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"Model and processor loaded on {device}.")

source = [
    {"type": "text", "source": 'Cats lying in bed'},
    {"type": "image", "source": "https://farm1.staticflickr.com/17/20770643_d04d79280b_z.jpg"},
    {"type": "text", "source": 'What are cute animals to cuddle with?'},
    {"type": "text", "source": 'What do cats look like?'},
    {"type": "image", "source": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1280px-Cute_dog.jpg"}
]

def encode_image(imageurl):
    print(f"Encoding image: {imageurl}")
    response = requests.get(imageurl)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.get_image_features(**inputs)
    return outputs.detach().cpu().numpy()

def encode_text(text):
    print(f"Encoding text: {text}")
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.get_text_features(**inputs)
    return outputs.detach().cpu().numpy()

def search(query, num_results=5):
    print(f"Searching for: {query}")
    query_embedding = encode_text(query)
    _, indices = index.search(query_embedding[np.newaxis, :], num_results)
    return [source[idx]['source'] for idx in indices[0]]

print("Encoding source items...")
embeddings = []
for item in source:
    if item["type"] == 'image':
        embeddings.append(encode_image(item['source']))
    else:
        embeddings.append(encode_text(item['source']))

print("Stacking embeddings...")
embeddings = np.vstack(embeddings)

print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
print("Adding embeddings to index...")
index.add(embeddings.astype('float32'))

print("Performing search...")
results = search('Dog')
print("Search results:", results)