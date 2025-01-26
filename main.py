from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io

app = Flask(__name__)
CORS(app)

# Load the smaller BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-small")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-small")

def generate_image_caption(image):
    """
    Generate a caption for the given image using the BLIP model.
    """
    with torch.no_grad():  # Avoid unnecessary memory usage by disabling gradient calculations
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

    # Generate caption using the BLIP model
    caption = generate_image_caption(image)

    # Log the caption to the console (this should be a debug log)
    print(f"Generated Caption: {caption}")

    # Return the caption in JSON response
    return jsonify({"caption": caption})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
