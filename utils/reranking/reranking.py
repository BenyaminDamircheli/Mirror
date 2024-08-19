from sentence_transformers import CrossEncoder
import numpy as np

class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder("mixedbread-ai/mxbai-rerank-large-v1")

    def rerank(self, query, chroma_results):
        # return a list of similarity scores for the query and the documents
        scores = self.model.predict([[query, doc] for doc in chroma_results['documents'][0]])
        # get the max similarity score
        mscore_index = np.argmax(scores)

        # return dictionary containing all of the data about the top scoring document (doc, id, distance, metadata, etc).
        res = {}
        for key in chroma_results:
            if (chroma_results[key] is not None):
                res[key] = chroma_results[key][0][mscore_index]
        return res