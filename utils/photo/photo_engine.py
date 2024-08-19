import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration


class PhotoEngine:
    def __init__(self, model:str):
        if model == "default":
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        else:
            self.model = BlipForConditionalGeneration.from_pretrained(model)
            self.processor = BlipProcessor.from_pretrained(model)

    
    def open_image(self, img_url:str):
        print("Opening image from URL:", img_url)
        response = requests.get(img_url, stream=True)  
        opened_image = BytesIO(response.content)
        opened_image = Image.open(opened_image).convert('RGB')
        print("Image Opened")
        return opened_image
    
    def describe_image(self, opened_image) -> str:
        print("Describing image")
        inputs = self.processor(opened_image, return_tensors="pt")
        out = self.model.generate(**inputs)
        print("Description generated")
        return self.processor.decode(out[0], skip_special_tokens=True)



# Test usage

# engine = PhotoEngine("default")
# opened_image = engine.open_image("https://plus.unsplash.com/premium_photo-1670745084868-7b4f727cc934?q=80&w=2864&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
# print(engine.describe_image(opened_image))
# output: several hands are shown with henna designs on them in a circle
# pretty good!
