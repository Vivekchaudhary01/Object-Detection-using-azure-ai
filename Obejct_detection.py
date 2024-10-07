import streamlit as st
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw
import numpy as np

# Streamlit UI
st.title("Azure Custom Vision Object Detection")

# Replace these with your actual Azure Custom Vision details
project_id = 'your project ID' # Replace with your project ID
cv_key = 'your resource primary key' # Replace with your prediction resource primary key
cv_endpoint = 'your resource endpoint' # Replace with your prediction resource endpoint

model_name = 'detect-produce'  # Replace with your actual model name
st.write(f'Ready to predict using model **{model_name}** in project **{project_id}**')

# Allow users to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    test_img = Image.open(uploaded_file)
    test_img_w, test_img_h = test_img.size

    # Display the uploaded image
    st.image(test_img, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")

    # Get a prediction client for the object detection model
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": cv_key})
    predictor = CustomVisionPredictionClient(endpoint=cv_endpoint, credentials=credentials)

    # Ensure the image stream is passed correctly as binary
    uploaded_file.seek(0)  # Reset the file pointer to the start
    image_data = uploaded_file.read()  # Read the file as binary

    try:
        # Detect objects in the uploaded image
        results = predictor.detect_image(project_id, model_name, image_data)

        # Draw the results on the image
        draw = ImageDraw.Draw(test_img)
        lineWidth = int(test_img_w / 100)
        object_colors = {
            "apple": "lightgreen",
            "banana": "yellow",
            "orange": "orange"
        }

        for prediction in results.predictions:
            if (prediction.probability * 100) > 50:  # Threshold
                color = object_colors.get(prediction.tag_name, 'white')
                left = prediction.bounding_box.left * test_img_w
                top = prediction.bounding_box.top * test_img_h
                height = prediction.bounding_box.height * test_img_h
                width = prediction.bounding_box.width * test_img_w
                points = ((left, top), (left + width, top), (left + width, top + height), (left, top + height), (left, top))
                draw.line(points, fill=color, width=lineWidth)
                st.text(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")

        # Display the result image with bounding boxes
        st.image(test_img, caption="Processed Image", use_column_width=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
