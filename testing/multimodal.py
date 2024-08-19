# After playing with this, I have come to the conclusion that either I am doing smth wrong
# or the embeddings suck. Probably the former.

import torch
import io
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests

print("Loading processor...")
processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
print("Loading model...")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
print("Loading text model...")
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
# set text model to eval mode
# better for inference and testing as it removes dropout and batchnorm
text_model.eval()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # basically takes the sum of the embeddings weighted by the attention mask (what tokens are important)
    # and divides by the sum of the attention mask (number of non-padding tokens)
    # Gives average value of the embeddings weighted by the attention mask as a new average vector representing the input sequence.
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_images(imageurl):
    print("Embedding Image")
    image = Image.open(requests.get(imageurl, stream=True).raw)
    image_features = processor(image, return_tensors="pt")
    image_emb = vision_model(**image_features).last_hidden_state
    image_emb_norm = F.normalize(image_emb[:, 0], p=2, dim=1)
    return image_emb_norm

def embed_text(text):
    print("Embedding Text")
    encoded_input = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        model_output = text_model(**encoded_input)

    # returns a vector that is representative of the whole input text.
    text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    text_embeddings = F.layer_norm(text_embeddings, normalized_shape=(text_embeddings.shape[1],))
    text_emb_norm  = F.normalize(text_embeddings, p=2, dim=1)
    return text_emb_norm

def distributor(type, source):
    if type == "image":
        return embed_images(source)
    elif type == "text":
        return embed_text(source)
    else:
        raise ValueError("Invalid type")


def similarity_search(img_embs, text_embs, sources):
    print("Performing Similarity Search")
    similarity_scores = torch.matmul(img_embs, text_embs.T)
    print(f"Similarity Scores: {similarity_scores}")
    sorted_scores, sorted_index = torch.sort(similarity_scores, descending=True)
    for idx in sorted_index[0]:
        source_type = sources[idx]['type']
        source_content = sources[idx]['source']
        print(f"Score: {sorted_scores[0][idx]}, Source Type: {source_type}, Source Content: {source_content}")


print("STARTING")
source = [
    {
        "type": "text",
        "source": 'Cats lying in bed'
    },
    {
        "type": "image",
        "source": "https://farm1.staticflickr.com/17/20770643_d04d79280b_z.jpg"
    },
    {
        "type": "text",
        "source": 'What are cute animals to cuddle with?'
    },
    {
        "type": "text",
        "source": 'What do cats look like?'
    },
        {
        "type": "image",
        "source": "https://images.unsplash.com/photo-1602886525304-4e9044f09157?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    }
]

query = distributor('text', 'Search Query: Cats')
# return list of embedded text and images
corpus = [distributor(item['type'], item['source']) for item in source]
corpus_embeddings = torch.cat(corpus, dim=0)
similarity_search(query, corpus_embeddings, source)

