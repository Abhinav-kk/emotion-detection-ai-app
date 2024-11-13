from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


def preprocess_image(image_path, target_size=(48, 48)):
    # Load the image
    image = load_img(image_path, color_mode='grayscale',
                     target_size=target_size)
    # Convert the image to array
    image = img_to_array(image)
    # Normalize the pixel values
    image = image / 255.0
    # Reshape the image to add batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def predict_emotion(image_path, model, emotion_list):
    # Preprocess the image
    image = preprocess_image(image_path)
    # Make prediction
    prediction = model.predict(image)
    # Decode the prediction
    # return prediction
    predicted_emotion = emotion_list[np.argmax(prediction)]
    return predicted_emotion


image_path = 'cropped_face.jpg'

# Predict the emotion
