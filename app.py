import streamlit as st
import ollama
from PIL import Image

def analyze_image(image_data):
    try:
        # Send image to Ollama for analysis
        response = ollama.chat(
            model='llama3.2-vision',  # Update model name as needed
            messages=[
                {
                    'role': 'user',
                    'content': 'What is in this image?',
                    'images': [image_data],
                },
            ],
        )
        return response['message']['content']
    except Exception as e:
        return f"Error processing image: {str(e)}"

def main():
    st.title("Image Analysis App")
    st.write("Welcome! Please upload an image, and I'll analyze it for you.")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert image to bytes
        image_data = uploaded_file.read()

        # Analyze the image
        st.write("Analyzing the image...")
        result = analyze_image(image_data)

        # Display analysis result
        st.write("Analysis Result:")
        st.write(result)
    else:
        st.write("Please upload an image to analyze.")

if __name__ == "__main__":
    main()
