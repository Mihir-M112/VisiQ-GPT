VisiQ-GPT: "Vision" and "IQ" Suggesting a system that provides intelligent answers using visual data with the integration of GPT

This project using llama3.2 vision for image processing and for general Q/A also.
Project name is VisiQ-GPT.
It should:
1. Implemented using ollama locally.
2. It is RAG based. For embeddings i will be using Nomic embed text also from ollama.
3. Document Upload and Indexing: Upload PDFs and images, which are then indexed using ColPali for retrieval.
4. Chat Interface: Engage in a conversational interface to ask questions about the uploaded documents. 
5. Session Management: Create, rename, switch between, and delete chat sessions.
6. Persistent Indexes: Indexes are saved on disk and loaded upon application restart.
7. It is integrated with MongoDB and Docker.
The model(llama3.2-vision) generate responses by understanding both the visual and textual content of the documents.