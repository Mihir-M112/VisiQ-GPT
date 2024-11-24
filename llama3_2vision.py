import ollama

with open ('landscape.jpg', 'rb') as f:
    
    response = ollama.chat(
        model='llama3.2-vision',
        messages=[{
            'role': 'user',
            'content': 'What is in this image?',
            'images': [f.read()]
        }]
    )

print(response['message']['content'])

# # Add project root to sys.path to import logger
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from logger import get_logger
