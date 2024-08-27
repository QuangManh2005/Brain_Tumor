import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('BrainTumor20EpochsCategorical.h5')

# Read and preprocess the image
image_path = 'D:\\Brain_Tumor\\pred\\pred9.jpg' # Đọc hình ảnh từ đường dẫn
image = cv2.imread(image_path)

# Convert image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to PIL Image and resize
img = Image.fromarray(image_rgb)
img = img.resize((64, 64))

# Convert image to numpy array and normalize
img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]

# Expand dimensions to match the model input shape
input_img = np.expand_dims(img, axis=0)

# Make predictions
predictions = model.predict(input_img)

# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)

print(predicted_class)
