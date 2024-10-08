import chromadb
from chromadb.config import Settings, DEFAULT_DATABASE, DEFAULT_TENANT
import os
import sys
import numpy as np



from utils.embeddings.embeddings_engine import Embeddings_Engine

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)

class SearchEngine:
    def __init__(self):
        self.embeddings_engine = Embeddings_Engine("default")
        self.client = chromadb.PersistentClient(
            settings = Settings(),
            database= DEFAULT_DATABASE,
            tenant= DEFAULT_TENANT,
        )
        try:
            self.collection = self.client.get_collection("mirror_db")
        except:
            self.create_collection("mirror_db")
    
    def create_collection(self, collection_name: str):
        self.collection = self.client.create_collection(collection_name, metadata={"hnswspace": "cosine"})

    
    def add(self, concatenated, citation, filepath):
        embedding = self.embeddings_engine.embed(concatenated).tolist()
        id_str = f"{filepath}::{citation}"
        self.collection.add(
            documents=[concatenated],
            ids=[f"{id_str}"],
            embeddings=[embedding],
            metadatas=[{"filepath": filepath}]
        )

    def search(self, query_string, query_type):
        embedding = self.embeddings_engine.embed(query_string).tolist()

        if query_type != "":
            result = self.collection.query(
                query_embeddings=[embedding],
                include=["documents", "metadatas"],
                n_results=3
            )
        cleaned_result = {}
        for i in range(len(result["ids"][0])):
            cleaned_result[result["ids"][0][i]] = result["documents"][0][i]
        return cleaned_result
        

        
    def delete(self, collection_name: str):
        self.client.delete_collection(collection_name)
        print("successfully deleted the collection")

    def exists_in_colection(self, document_id:str):
        documents = self.collection.get(ids=[document_id])
        print(documents)
        return len(documents.get("ids")) > 0

        
        
        