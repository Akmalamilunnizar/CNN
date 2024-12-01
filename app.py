from flask import Flask, request, jsonify, url_for
import os
import numpy as np
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = './static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model('./cnn_koi_penyakit/trained_model.keras')
class_name = ['Bacterial Disease', 'Fungal Disease', 'Parasitic Disease']

@app.route('/predict', methods=['POST'])
def predict():
    # Save uploaded image
    if 'imagefile' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    imagefile = request.files['imagefile']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    # Preprocess the image for prediction
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch
    prediction = model.predict(input_arr)

    # Calculate probabilities
    probabilities = prediction.flatten() * 100
    percentages = np.round(probabilities, 2)
    result_index = np.argmax(probabilities)

    # Prepare response
    response = {
        'prediction': class_name[result_index],
        'probabilities': dict(zip(class_name, percentages.tolist())),
        'image_url': request.host_url + 'static/uploads/' + imagefile.filename
    }
    return jsonify(response)

@app.route('/')
def home():
    return 'Welcome to PCV API'

if __name__ == "__main__":
    # app.run(debug=True)
    app.run()
    
