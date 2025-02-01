
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os
import requests

app = Flask(__name__)
CORS(app)

def generate_image_caption(image):
    """
    Generate a caption for the given image using the Hugging Face API.
    """
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {os.environ['HUGGING_FACE_TOKEN']}"}
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    response = requests.post(API_URL, headers=headers, data=img_byte_arr)
    return response.json()[0]['generated_text']

@app.route('/caption', methods=['POST'])
def generate_caption():
    # Check if the image is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the image from the request
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    try:
        # Generate caption using Hugging Face API
        caption = generate_image_caption(image)
        print(f"Generated Caption: {caption}")
        return jsonify({"caption": caption})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
