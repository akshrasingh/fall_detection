import os
import tempfile
import imghdr
import base64
from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Define the UPLOAD_FOLDER as the 'uploads' directory within the same directory as the Flask application file
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
print(UPLOAD_FOLDER)
# Ensure the UPLOAD_FOLDER directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your pretrained model
model_path = os.path.join(os.path.dirname(__file__), 'my_model.h5')
my_model = load_model(model_path)
print(my_model)
# Preprocess the image for the model


def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Perform prediction using the model


def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = preprocess_image(img)
    prediction = my_model.predict(img)
    return prediction

# Home route


@app.route('/')
def home():
    return render_template('index.html')

# Models route


@app.route('/models')
def models():
    return render_template('models.html')

# Predict route


@app.route('/predict', methods=['POST'])
def predict_image():

    if request.method == 'POST':
        uploaded_file = request.files['image']

        if uploaded_file.filename != '':
            try:

                # Save the uploaded file to the 'uploads' directory
                filename = secure_filename(uploaded_file.filename)
                uploaded_file.save(os.path.join(
                    app.config['UPLOAD_FOLDER'], filename))
                image_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], filename)

                # Perform prediction
                prediction = predict(image_path)

                predicted_label = "Fall" if prediction < 0.5 else "No Fall"

                # Read the saved image file and convert it to base64 encoding
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(
                        image_file.read()).decode('utf-8')

                # Determine the image type
                image_type = imghdr.what(image_path)
                if image_type not in ['jpeg', 'jpg', 'png', 'gif']:
                    raise ValueError("Unsupported image format")

                # Render the result page with base64 encoded image data
                return render_template('result.html', prediction=predicted_label, image_data=image_data, image_type=image_type)

            except Exception as e:
                flash('Error occurred during prediction: {}'.format(str(e)))
                return redirect('/result')
        else:
            flash('Please upload an image file.')
            return redirect('/result')


@app.route('/result')
def result():
    prediction = request.args.get('prediction', '')
    image_name = request.args.get('image_name', '')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)

    return render_template('result.html', prediction=prediction, image_path=image_path)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
