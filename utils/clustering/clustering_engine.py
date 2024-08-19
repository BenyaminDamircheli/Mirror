from sentence_transformers import SentenceTransformer, util
import numpy as np
from utils.embeddings.embeddings_engine import Embeddings_Engine

class Clustering_Engine:
    def __init__(self, embeddings_engine: Embeddings_Engine, threshold = 0.7):
        self.embeddings_engine = embeddings_engine
        self.threshold = threshold
        self.documents = []
        self.document_embeddings = []


        def add_document(self, document:str):
            new_embedding = self.embeddings_engine.embed(document)[0]

            if self.document_embeddings:
                similarities = [float(util.cos_sim(new_embedding, embedding)[0][0]) for embedding in self.document_embeddings]
                max_similarity = max(similarities)
                best_match_index = np.argmax(similarities)

                if max_similarity >= threshold:
                    self.documents[best_match_index].append(document)
                    self.document_embeddings[best_match_index] = np.mean(
                        [self.document_embeddings[best_match_index], new_embedding], axis=0
                    )
                    return


            # if no similar documents or document clusters don't exist yet, create new cluster
            # with just the new document (list of list of documents).
            self.documents.append([document])
            self.document_embeddings.append(new_embedding)

        def get_documents(self):
            return self.documents

