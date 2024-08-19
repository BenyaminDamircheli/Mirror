from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import torch
import os
import cv2
import tempfile
import moviepy.editor as mp
from PyPDF2 import PdfReader
from PIL import Image

from utils.summary.summary_engine import SummaryEngine
from utils.audio.audio_engine import AudioEngine
from utils.clustering.clustering_engine import Clustering_Engine
from utils.photo.photo_engine import PhotoEngine
from utils.reranking.reranking import Reranker
from utils.search.search_engine import SearchEngine
from utils.embeddings.embeddings_engine import Embeddings_Engine

class Engine: 
    def __init__(self): 
        self.audio_engine = AudioEngine("default")
        # self.clustering_engine = Clustering_Engine()
        self.embeddings_engine = Embeddings_Engine("default")
        self.photo_engine = PhotoEngine("default")
        self.reranking = Reranker()
        self.search_engine = SearchEngine()
        self.summary_engine = SummaryEngine()




