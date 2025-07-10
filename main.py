import streamlit as st
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import saving # Import saving module
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd # Import pandas

st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide")

# Removed the class_names list as requested.

@st.cache_data
def load_data():
    """Loads and preprocesses the Fashion MNIST dataset."""
    # We still load data for sample image display and dataset insights
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape((60000, 784)).astype('float32') / 255
    x_test = x_test.reshape((10000, 784)).astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

@st.cache_resource
def load_pretrained_model():
    """Loads the pre-trained Keras model."""
    try:
        # Load back your model exactly as it was
        loaded_model = saving.load_model('fashion_mnist_mlp.keras')
        st.success("Pre-trained model 'fashion_mnist_mlp.keras' loaded successfully!")
        return loaded_model
    except Exception as e:
        st.error(f"Error loading pre-trained model: {e}")
        st.info("Please ensure 'fashion_mnist_mlp.keras' is in the same directory as this script.")
        return None

# --- Main App Layout ---
st.title("Fashion MNIST Image Classifier 👗👕👟")

st.markdown("""
Welcome to the Fashion MNIST Classifier! This app uses a pre-trained neural network model to identify different types of clothing items from images.
You can explore sample images and upload your own images for prediction.
""")

# Load data once
if 'x_train' not in st.session_state:
    (st.session_state.x_train, st.session_state.y_train), (st.session_state.x_test, st.session_state.y_test) = load_data()

# Load the pre-trained model
st.session_state.model = load_pretrained_model()

# Image Prediction Section
st.header("1. Predict on Images")

if st.session_state.model is None:
    st.warning("Model could not be loaded. Please check the error message above.")
else:
    tab1, tab2 = st.tabs(["Sample Image Prediction", "Upload Your Image"])

    with tab1:
        st.subheader("Predict on a Sample Image")
        sample_idx = st.number_input(
            'Enter an image index (0-9999 from test set)',
            min_value=0, max_value=len(st.session_state.x_test) - 1, value=0
        )
        img_display = st.session_state.x_test[sample_idx].reshape(28, 28)
        # Display original label as number
        st.image(img_display, width=150, caption=f"Original Label: {st.session_state.y_test[sample_idx]}")

        if st.button("Predict Sample Image"):
            img_to_predict = st.session_state.x_test[sample_idx].reshape(1, 784)
            pred_probs = st.session_state.model.predict(img_to_predict)[0]
            predicted_label_index = np.argmax(pred_probs)
            confidence = pred_probs[predicted_label_index] * 100

            # Display predicted label as number
            st.success(f"**Predicted: {predicted_label_index}** (Confidence: {confidence:.2f}%)")

            # Create a DataFrame for the bar chart, using numerical indices for x-axis
            df_pred = pd.DataFrame({'Probability': pred_probs})
            st.bar_chart(df_pred)


    with tab2:
        st.subheader("Upload Your Own Image for Prediction")
        uploaded_file = st.file_uploader("Choose an image (Grayscale preferred, will be resized to 28x28 pixels)", type=['png', 'jpg', 'jpeg'])

        if uploaded_file is not None:
            try:
                # Open, convert to grayscale, and resize
                image = Image.open(uploaded_file).convert('L').resize((28, 28))
                arr = np.array(image).astype('float32') / 255

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, caption='Uploaded Image', width=150)
                with col2:
                    arr_reshaped = arr.reshape((1, 784))
                    pred_probs = st.session_state.model.predict(arr_reshaped)[0]
                    predicted_label_index = np.argmax(pred_probs)
                    confidence = pred_probs[predicted_label_index] * 100

                    # Display predicted label as number
                    st.success(f"**Predicted: {predicted_label_index}** (Confidence: {confidence:.2f}%)")

                    # Create a DataFrame for the bar chart, using numerical indices for x-axis
                    df_pred = pd.DataFrame({'Probability': pred_probs})
                    st.bar_chart(df_pred)

            except Exception as e:
                st.error(f"Error processing image: {e}. Please ensure it's a valid image file.")

st.markdown("---")

# Dataset Statistics
st.header("2. Dataset Insights") # Changed header number
st.markdown("Understand the distribution of different clothing items in the training dataset.")
if st.checkbox('Show Fashion MNIST Class Distribution'):
    labels, counts = np.unique(st.session_state.y_train, return_counts=True)
    fig_dist, ax_dist = plt.subplots()
    # Use numerical labels for x-axis in matplotlib chart
    ax_dist.bar(labels, counts)
    ax_dist.set_title('Fashion MNIST Training Set Class Distribution')
    ax_dist.set_xlabel('Class Label (Number)')
    ax_dist.set_ylabel('Number of Images')
    ax_dist.set_xticks(labels) # Ensure all numerical labels are shown
    st.pyplot(fig_dist)