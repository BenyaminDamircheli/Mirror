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
        print("Initializing Audio Engine")
        self.audio_engine = AudioEngine()
        # self.clustering_engine = Clustering_Engine()
        print("Initializing Embeddings Engine")
        self.embeddings_engine = Embeddings_Engine("default")
        print("Initializing Photo Engine")
        self.photo_engine = PhotoEngine("default")
        print("Initializing Reranking Engine")
        self.reranking = Reranker()
        print("Initializing Search Engine")
        self.search_engine = SearchEngine()
        print("Initializing Summary Engine")
        self.summary_engine = SummaryEngine()

        self.video_fragments_dir = "store"
        os.makedirs(self.video_fragments_dir, exist_ok=True)
        self.interval = 60
        
        self.csv_filename = "extracted.csv"
        self.load_existing_videos()

    def load_existing_videos(self):
        if os.path.exists(self.csv_filename):
            data = pd.read_csv(self.csv_filename)
            video_ids = data['Video_ID'].tolist()
            self.existing_videos = {video_id: 1 for video_id in video_ids}
        else:
            self.existing_videos = {}
    
    def extract_frames(self, video_path, interval, fps):
        print("Extracting frames from video")
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        seconds_count = 0
        while True:
            returned, frame = cap.read()
            if not returned:
                break
            if frame_count == int(seconds_count * fps):
                frames.append(frame)
                seconds_count += interval
            frame_count += 1
        cap.release()
        return frames
    
    def process_video(self, video_path, filename):
        video_clip = mp.VideoFileClip(video_path)
        fps = video_clip.fps
        frames = self.extract_frames(video_path, self.interval, fps)

        for seconds, frame in enumerate(frames):
            video_id = f"{video_path}::{seconds * self.interval}"
            if self.search_engine.exists_in_colection(video_id):
                print(f"Skipping already indexed frame: {video_id}")
                continue
            description = self.photo_engine.describe_image(frame)
            start_time = seconds * self.interval
            duration = self.interval
            end_time = min(start_time + duration, video_clip.duration)

            if end_time > start_time:
                video_fragment = video_clip.subclip(start_time, end_time)
                with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as temp_audio_file:
                    video_fragment.audio.write_audiofile(temp_audio_file.name)
                    transcript = self.audio_engine.transcribe_audio(temp_audio_file.name)
                    summary = self.summary_engine.summarize(transcript)
                concat_description = f"Title: {filename} \n {description} \n {summary}"
                self.search_engine.add(concat_description, start_time, video_path)
    
    def process_pdf(self, path, filename):
        pdf = PdfReader(path)

        for idx, page in enumerate(pdf.pages):
            page_content = page.extract_text()
            self.search_engine.add(page_content, idx, path)

    def process_image(self, path, filename):
        with Image.open(path) as img:
            description = self.photo_engine.describe_image(img)
            concat = f"Title: {filename} \n {description}"
            self.search_engine.add(concat, 0, path)

    def process_text(self, path,filename):
        with open(path, "r") as file:
            text = file.read()
        
        text_words = text.split()
        text_groups = [text_words[i:i+100] for i in range(0, len(text_words), 100)]

        for idx, group in enumerate(text_groups):
            group_text = " ".join(group)
            print(group_text)
            if group_text:
                concat = f"Title: {filename} \n {group_text}"
                self.search_engine.add(concat, idx, path)
    
    def process_all_files(self, path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".pdf"):
                    self.process_pdf(file_path, file)
                elif file.endswith((".txt", ".md")):
                    self.process_text(file_path, file)
                elif file.endswith((".jpg", ".png", ".jpeg")):
                    self.process_image(file_path, file)
                elif file.endswith(".mp4"):
                    self.process_video(file_path, file)
                else:
                    print(f"Skipping unsupported file type: {file_path}")

engine = Engine()
engine.process_all_files("input")
