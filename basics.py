from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)








from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your custom model
model_path = r"C:\Users\Legion\datasets\cnn_koi_penyakit\trained_model.h5"  # Use a raw string for Windows paths
model = load_model(model_path)

# Define your class names (replace with the actual classes of your model)
class_names = ['Bacterial Disease', 'Fungal Disease', 'Parasitic Disease']  # Example class names

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    # Save uploaded image
    imagefile = request.files['imagefile']
    image_path = "./test/" + imagefile.filename
    imagefile.save(image_path)

    # Preprocess the image
    image = load_img(image_path, target_size=(128, 128))  # Match your model's input size

    image = img_to_array(image)
    image = image / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)

    # Predict using the custom model
    yhat = model.predict(image)
    print("Predictions:", yhat)

    predicted_class = np.argmax(yhat, axis=-1)  # Get the class index with the highest probability
    classification = f"{class_names[predicted_class[0]]} ({yhat[0][predicted_class[0]] * 100:.2f}%)"
# Provide the image URL for rendering
    image_path = r"C:\Users\Legion\datasets\cnn_koi_penyakit\trained_model.h5"  # Use a raw string for Windows paths
    image_url = f"/static/{imagefile.filename}"

    return render_template('index.html', prediction=classification, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
