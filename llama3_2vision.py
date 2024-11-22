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