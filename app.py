import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

# Load the trained model
model = load_model('BrainTumor20EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_class_name(class_no):
    """ Convert numeric class to a human-readable string """
    return "Khong Co U Nao" if class_no == 0 else "Co U Nao"

def get_result(img_path):
    """ Preprocess the image and make a prediction """
    # Read and preprocess the image
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image_pil = Image.fromarray(image_rgb)
    image_resized = image_pil.resize((64, 64))
    image_array = np.array(image_resized) / 255.0  # Normalize pixel values to [0, 1]
    input_img = np.expand_dims(image_array, axis=0)
    
    # Make predictions
    predictions = model.predict(input_img)
    result = np.argmax(predictions, axis=1)[0]  # Get the predicted class
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Ensure the 'uploads' directory exists
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        
        # Get result from the uploaded image
        class_no = get_result(file_path)
        result = get_class_name(class_no) 
        return result

    return None

if __name__ == '__main__':
    app.run(debug=True)
