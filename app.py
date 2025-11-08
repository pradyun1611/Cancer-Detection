import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

MODEL_CONFIGS = {
    "Kidney": {
        "path": "./models/kindey_cancer.h5",
        "image_size": (128, 128),
        "class_names": ["Normal", "Tumor"],
        "model_type": "binary"
    },
    "Chest": {
        "path": "./models/chest_cancer.h5",
        "image_size": (128, 128),
        "class_names": ['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma'],
        "model_type": "multiclass"
    },
    "Cervical": {
        "path": "./models/cervical_cancer.h5",
        "image_size": (128, 128),
        "class_names": ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediat'],
        "model_type": "multiclass"
    },
    "Oral": {
        "path": "./models/oral_cancer.h5",
        "image_size": (128, 128),  
        "class_names": ["Normal", "Squamous Cell Carcinoma"],
        "model_type": "binary"  
    }
}

@st.cache_resource
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}.\nEnsure the file is in the correct path.\nError: {e}")
        return None

def preprocess_image(image_file, target_size):
    try:
        image = Image.open(image_file)
        
        image = image.convert('RGB')
        
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        img_array = np.asarray(image)
        
        img_array = img_array[:, :, ::-1]
        
        img_array = img_array / 255.0
        
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def interpret_prediction(prediction, class_names, model_type):
    try:
        if model_type == 'binary':
            score = prediction[0][0]
            
            if score > 0.5:
                predicted_class = class_names[1]
                confidence = score * 100
            else:
                predicted_class = class_names[0]
                confidence = (1 - score) * 100
            
            return predicted_class, confidence
        
        elif model_type == 'multiclass':
            prediction_array = prediction[0]
            predicted_class_index = np.argmax(prediction_array)
            predicted_class = class_names[predicted_class_index]
            confidence = prediction_array[predicted_class_index] * 100
            
            return predicted_class, confidence
        
        else:
            st.error(f"Unknown model_type: {model_type}. Must be 'binary' or 'multiclass'.")
            return None, None

    except Exception as e:
        st.error(f"Error interpreting prediction: {e}.")
        return None, None

st.set_page_config(page_title="Cancer Detection Portal", layout="wide")

st.title("🔬 AI Cancer Detection Portal")
st.markdown("Select a cancer type and upload a scan to get a prediction.")

st.sidebar.header("Controls")

cancer_type = st.sidebar.selectbox(
    "Select Cancer Type:",
    list(MODEL_CONFIGS.keys())
)

uploaded_file = st.sidebar.file_uploader(
    f"Upload {cancer_type} Scan Image", 
    type=["jpg", "jpeg", "png", "bmp", "tif"]
)

col1, col2 = st.columns(2)

with col1:
    st.header("Uploaded Scan")
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded {cancer_type} Scan", use_container_width=True)
        except Exception as e:
            st.error(f"Could not display image. Error: {e}")
    else:
        st.info("Please upload an image using the sidebar.")
        

with col2:
    st.header("Prediction Result")
    if uploaded_file is not None and cancer_type:
        
        model_config = MODEL_CONFIGS[cancer_type]
        model_path = model_config['path']
        image_size = model_config['image_size']
        class_names = model_config['class_names']
        model_type = model_config['model_type']
        
        with st.spinner(f"Loading {cancer_type} model..."):
            model = load_keras_model(model_path)
        
        if model:
            with st.spinner("Analyzing image..."):
                processed_image = preprocess_image(uploaded_file, image_size)
            
            if processed_image is not None:
                with st.spinner("Running prediction..."):
                    try:
                        prediction = model.predict(processed_image)
                        
                        predicted_class, confidence = interpret_prediction(prediction, class_names, model_type)
                        
                        if predicted_class is not None:
                            if cancer_type == "Cervical":
                                if predicted_class == class_names[0] or predicted_class == class_names[1]:
                                    st.error(f"**Result: {predicted_class}**")
                                else:
                                    st.success(f"**Result: {predicted_class}**")
                            else:
                                if predicted_class == class_names[0]:
                                    st.success(f"**Result: {predicted_class}**")
                                else:
                                    st.error(f"**Result: {predicted_class}**")
                            
                            st.metric(
                                label=f"Confidence in '{predicted_class}'",
                                value=f"{confidence:.2f}%"
                            )
                            
                            with st.expander("Raw Model Output"):
                                st.json({"raw_prediction": prediction.tolist()})

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")

                st.warning(
                    "**Disclaimer:** This is an AI-generated prediction and not a medical diagnosis. "
                    "Please consult a qualified healthcare professional for any medical concerns."
                )

    elif not uploaded_file:
        st.info("Waiting for image upload...")
    
st.sidebar.markdown("---")
st.sidebar.info(
    "This portal uses pre-trained CNN models to detect potential abnormalities. "
    "Results are not a substitute for professional medical advice."
)

