import os
from pathlib import Path
import logging

# Configure logging for better visibility of actions
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files required in the multi-modal retrieval project
list_of_files = [
    "src/__init__.py",                # Initialization file for the source package
    "src/data_preprocessing.py",       # File for data preprocessing steps (e.g., downloading and parsing Wikipedia data)
    "src/vector_store.py",             # File for vector store setup (e.g., Qdrant vector store setup)
    "src/embedding_models.py",         # File for loading and using embedding models (e.g., GPT, CLIP)
    "src/retrieval_system.py",         # File for implementing the retrieval logic (e.g., querying the multi-modal index)
    "src/utils.py",                    # File for utility functions like plotting images
    ".env",                            # Environment file for storing sensitive data like API keys
    "setup.py",                        # Setup file for packaging the project
    "research/experiments.ipynb",      # Jupyter notebook for experiments with different queries and retrievals
    "tests/test_vector_store.py",      # Test case for vector store functionality
    "tests/test_embedding_models.py",  # Test case for embedding models
    "tests/test_retrieval_system.py",  # Test case for retrieval logic
    "tests/test_utils.py",             # Test case for utility functions
    "app.py",                          # Entry point for the application or API
    "store_index.py",                  # Script for managing index storage and retrieval
    "static/.gitkeep",                 # Keeps the static folder in Git (for CSS, JS)
    "templates/index.html"             # HTML template for the web interface
]

# Loop over the list of files and create the necessary directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directories if they don't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Create the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass  # Create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists and is not empty")
