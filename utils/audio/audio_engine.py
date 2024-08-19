import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor



class AudioEngine:
    def __init__(self, model:str):
        if model == "default":
            self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
            self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(model)
            self.processor = WhisperProcessor.from_pretrained(model)
        

    def transcribe_audio(self, path:str):
        full_text = ""
        segements, info = self.model.transcribe(path)
        for segment in segements:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            full_text += segment["text"]
        
        return full_text

        