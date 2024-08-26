from faster_whisper import WhisperModel




class AudioEngine:
    def __init__(self, ):
        self.model = WhisperModel("medium.en")
        

    def transcribe_audio(self, path:str):
        full_text = ""
        segements, info = self.model.transcribe(path)
        for segment in segements:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            full_text += segment.text + " "
        
        return full_text

        