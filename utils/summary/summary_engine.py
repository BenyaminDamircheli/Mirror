from transformers import pipeline

class SummaryEngine:
    def __init__(self):
        self.model = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self, text: str):
        summary = self.model(text, max_length=60, min_length=30, do_sample=False)
        return summary



