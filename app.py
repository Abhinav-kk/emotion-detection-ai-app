from flask import Flask, render_template, request, url_for
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.models import load_model
import cv2
import os

# Define paths for the models
MODEL_PATHS = {
    "Scratch CNN Model": './ML_Models/emotion_detection_model_base_cnn.h5',
    "ImageNet Model": './ML_Models/emotion_detection_imagenet_model.h5',
    "MobileNet Model": './ML_Models/emotion_detection_mobilenet_model.h5',
    "VGG16 Model": './ML_Models/emotion_detection_vgg16_model.h5'
}


def extract_face(image_path, output_path="./static/cropped_face.jpg"):
    # Load the Haar cascade for face detection
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read the image in grayscale mode
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Could not read the image.")
        return "No face detected."

    # Save the grayscale image
    gray_output_path = "gray_" + output_path
    cv2.imwrite(gray_output_path, img)
    print(f"Grayscale image saved as {gray_output_path}")

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if a face is detected
    if len(faces) == 0:
        print("No face detected.")
        return "No face detected."

    # Extract the first detected face (assuming there's only one face)
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]

    # Resize the face to 48x48 pixels
    face_resized = cv2.resize(face, (48, 48))

    # Save the cropped and resized grayscale face
    cv2.imwrite(output_path, face_resized)
    print(f"Face extracted, resized to 48x48, and saved as {output_path}")


def preprocess_image(image_path, model_name, target_size=(48, 48)):
    # Load the image in grayscale
    image = load_img(image_path, color_mode='grayscale',
                     target_size=target_size)
    image = img_to_array(image)  # Convert to array
    image = image / 255.0  # Normalize pixel values

    # Repeat channels if model is not base_cnn
    if model_name != "Scratch CNN Model":
        image = np.repeat(image, 3, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def predict_emotion(image_path, model, model_name, emotion_list):
    image = preprocess_image(image_path, model_name)
    prediction = model.predict(image)
    predicted_emotion = emotion_list[np.argmax(prediction)]
    return predicted_emotion


def draw_bounding_box(image_path, output_path="./static/face_with_bounding_box.jpg"):
    # Load the Haar cascade for face detection
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read the image.")
        return "No face detected."

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if a face is detected
    if len(faces) == 0:
        print("No face detected.")
        return "No face detected."

    # Draw a bounding box around the first detected face
    x, y, w, h = faces[0]
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Save the image with the bounding box
    cv2.imwrite(output_path, img)
    print(f"Image with bounding box saved as {output_path}")
    return f"Image with bounding box saved as {output_path}"


app = Flask(__name__)
emotion_list = ['angry', 'disgust', 'fear',
                'happy', 'neutral', 'sad', 'surprise']


def getResult(selected_option, image):
    image.save("./static/image.jpg")
    model_path = MODEL_PATHS[selected_option]
    model = load_model(model_path)

    bounding_face_result = draw_bounding_box("./static/image.jpg")
    if bounding_face_result == "No face detected.":
        return bounding_face_result, [], None

    extract_face_result = extract_face(
        "./static/image.jpg", output_path="./static/cropped_face.jpg")
    if extract_face_result == "No face detected.":
        return extract_face_result, [], None

    image_path = './static/cropped_face.jpg'
    predicted_emotion = predict_emotion(
        image_path, model, selected_option, emotion_list)

    images = [url_for('static', filename='face_with_bounding_box.jpg'),
              url_for('static', filename='cropped_face.jpg')]

    return "The predicted emotion is:", images, predicted_emotion


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    images = []
    predicted_emotion = None

    dropdown_options = ["Select a model", "Scratch CNN Model", "ImageNet Model",
                        "MobileNet Model", "VGG16 Model"]

    if request.method == 'POST':
        selected_option = request.form.get('option')
        image = request.files.get('image')
        if selected_option != "Select a model":
            result, images, predicted_emotion = getResult(
                selected_option, image)
        return render_template('index.html', result=result, dropdown_options=dropdown_options, images=images, predicted_emotion=predicted_emotion, model_name=selected_option)

    return render_template('index.html', result=result, dropdown_options=dropdown_options, images=[], predicted_emotion=None, model_name="Select")


if __name__ == '__main__':
    app.run(debug=True)
