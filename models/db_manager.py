import os
from pymongo import MongoClient
from typing import Optional, Dict, List
import json
from datetime import datetime
import numpy as np
from urllib.parse import quote_plus
from utils.logger import get_logger
from bson import ObjectId

logger = get_logger(__name__)

class MongoDBManager:
    _instance = None
    _connection_string = None
    
    # Update these credentials
    DEFAULT_USERNAME = "visiq_gpt1"
    DEFAULT_PASSWORD = "VisiQ_GPT@1"
    DEFAULT_CLUSTER = "visiqgpt-db.a9tpc.mongodb.net"
    
    @classmethod
    def _build_connection_string(cls, username: str, password: str, cluster: str) -> str:
        """Build MongoDB connection string with escaped credentials"""
        escaped_username = quote_plus(username)
        escaped_password = quote_plus(password)
        return f"mongodb+srv://{escaped_username}:{escaped_password}@{cluster}/?retryWrites=true&w=majority"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'client'):
            self.client = None
            self.db = None
            # Try custom connection first, then build default
            self._connection_string = os.getenv('MONGODB_URI', 
                self._build_connection_string(
                    self.DEFAULT_USERNAME,
                    self.DEFAULT_PASSWORD,
                    self.DEFAULT_CLUSTER
                ))
            if self._connection_string:
                self.connect()

    @classmethod
    def set_connection_string(cls, username: str, password: str, cluster: str):
        """Allow users to set MongoDB connection credentials"""
        cls._connection_string = cls._build_connection_string(username, password, cluster)
        if cls._instance:
            cls._instance.connect()

    def connect(self):
        """Connect to MongoDB using the connection string"""
        try:
            if self._connection_string is None:
                raise ValueError("MongoDB connection string not provided")
            
            logger.info("Attempting to connect to MongoDB...")
            self.client = MongoClient(
                self._connection_string,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client.visiq_embeddings
            
            # Create indexes for faster queries
            self.db.image_embeddings.create_index([("image_path", 1)], background=True)
            self.db.pdf_embeddings.create_index([("pdf_path", 1)], background=True)
            self.db.image_embeddings.create_index([("timestamp", -1)], background=True)
            self.db.pdf_embeddings.create_index([("timestamp", -1)], background=True)
            
            logger.info("Successfully connected to MongoDB and created indexes")
            return True
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            return False

    def _check_connection(self) -> bool:
        """Check if database connection is active"""
        try:
            if self.client is None or self.db is None:
                return False
            # Test connection is alive with timeout
            self.client.admin.command('ping', serverSelectionTimeoutMS=2000)
            return True
        except Exception as e:
            logger.error(f"MongoDB connection check failed: {e}")
            return False

    def store_pdf_embeddings(self, pdf_path: str, embeddings_data: Dict):
        """Store PDF embeddings in MongoDB"""
        if not self._check_connection():
            return None

        collection = self.db.pdf_embeddings
        
        # Convert numpy arrays to lists if present
        if isinstance(embeddings_data.get('embeddings'), np.ndarray):
            embeddings_data['embeddings'] = embeddings_data['embeddings'].tolist()

        document = {
            'pdf_path': pdf_path,
            'embeddings': embeddings_data.get('embeddings'),
            'metadata': embeddings_data.get('metadata', {}),
            'timestamp': datetime.now(),
            'chunks': embeddings_data.get('chunks', [])
        }

        return collection.update_one(
            {'pdf_path': pdf_path},
            {'$set': document},
            upsert=True
        )

    def store_image_embeddings(self, image_path: str, image_data: Dict):
        """Store image embeddings and analysis in MongoDB"""
        if not self._check_connection():
            return None

        collection = self.db.image_embeddings
        
        # Convert numpy arrays to lists if present
        if isinstance(image_data.get('embeddings'), np.ndarray):
            image_data['embeddings'] = image_data['embeddings'].tolist()

        document = {
            'image_path': image_path,
            'base64_image': image_data.get('base64_image'),
            'vision_analysis': image_data.get('vision_analysis'),
            'embeddings': image_data.get('embeddings'),
            'timestamp': datetime.now()
        }

        return collection.update_one(
            {'image_path': image_path},
            {'$set': document},
            upsert=True
        )

    def get_pdf_embeddings(self, pdf_path: str) -> Optional[Dict]:
        """Retrieve PDF embeddings from MongoDB"""
        if not self._check_connection():
            return None

        result = self.db.pdf_embeddings.find_one({'pdf_path': pdf_path})
        return result

    def get_image_embeddings(self, image_path: str) -> Optional[Dict]:
        """Retrieve image embeddings from MongoDB"""
        if not self._check_connection():
            return None

        result = self.db.image_embeddings.find_one({'image_path': image_path})
        return result

    def create_session(self) -> str:
        """Create a new chat session and return session ID"""
        if not self._check_connection():
            return None
            
        session = {
            'session_id': str(ObjectId()),
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'history': [],
            'current_file': None,
            'file_type': None
        }
        
        self.db.chat_sessions.insert_one(session)
        return session['session_id']

    def update_session_file(self, session_id: str, file_path: str, file_type: str):
        """Update the file associated with a session"""
        if not self._check_connection():
            return None
            
        self.db.chat_sessions.update_one(
            {'session_id': session_id},
            {
                '$set': {
                    'last_activity': datetime.now(),
                    'current_file': file_path,
                    'file_type': file_type
                }
            }
        )

    def get_session_file(self, session_id: str) -> dict:
        """Get the last file used in the session"""
        if not self._check_connection():
            return None
            
        session = self.db.chat_sessions.find_one({'session_id': session_id})
        if session and session.get('current_file'):
            return {
                'file_path': session['current_file'],
                'file_type': session['file_type']
            }
        return None

    def add_conversation_history(self, session_id: str, prompt: str, response: str):
        """Add a conversation to the session history, maintaining only last 5"""
        if not self._check_connection():
            return None
            
        self.db.chat_sessions.update_one(
            {'session_id': session_id},
            {
                '$push': {
                    'history': {
                        '$each': [{
                            'prompt': prompt,
                            'response': response,
                            'timestamp': datetime.now()
                        }],
                        '$slice': -5  # Keep only last 5 conversations
                    }
                },
                '$set': {'last_activity': datetime.now()}
            }
        )

    def get_conversation_history(self, session_id: str):
        """Get conversation history for a session"""
        if not self._check_connection():
            return []
            
        session = self.db.chat_sessions.find_one({'session_id': session_id})
        return session.get('history', []) if session else []