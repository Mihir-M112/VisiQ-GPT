from models.model_loader import ensure_model_installed, generate_response

# Ensure the model is installed
ensure_model_installed("llama3.2-vision")

# Test with a simple prompt (text-only)
prompt = "What is the capital of France?"
response = generate_response("llama3.2-vision", prompt)
print("Text-only Response:", response)

# # Test with an image prompt (replace with a valid image path)
# image_path = "landscape.jpg"  # Replace with your image file path
# response_with_image = generate_response("llama3.2-vision", prompt, image_path)
# print("Image-based Response:", response_with_image)
