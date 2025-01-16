from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

app = Flask(__name__)
CORS(app)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_caption(image):
    """
    Generate a caption for the given image using the BLIP model.
    """
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

@app.route('/caption', methods=['POST'])
def generate_caption():
    # Check if the image is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the image from the request
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    # Generate caption using BLIP model
    caption = generate_image_caption(image)
    print(f"Generated Caption: {caption}")
    return jsonify({"caption": caption})

# handler for Vercel
def handler(request, context):
    return app(request.environ, start_response=context.start_response)
