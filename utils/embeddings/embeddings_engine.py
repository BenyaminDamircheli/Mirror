from sentence_transformers import SentenceTransformer


class Embeddings_Engine:
    def __init__(self, model:str):
        if model == "default":
            self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
        else:
            self.model = SentenceTransformer(model, trust_remote_code=True)

    
    # COME BACK TO THIS LATER
        # self.text_tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5')
        # self.text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
        # self.vision_processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        # self.vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

    # Geneterate embeddings for text
    def embed(self, text: list[str]):
        embeddings = self.model.encode(text)
        return embeddings