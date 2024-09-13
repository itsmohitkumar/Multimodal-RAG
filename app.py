from flask import Flask, request, render_template, jsonify, send_file
import os
import requests
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import warnings
from pathlib import Path
import gdown
import ssl
import subprocess
import qdrant_client
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.schema import ImageNode

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Payload indexes have no effect in the local Qdrant.")

# Initialize Flask app
app = Flask(__name__)

# Setup environment and variables
def setup_environment():
    """Load environment variables and create necessary directories."""
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI API key not found. Set it in the environment variables or in a .env file.")
    input_image_path = Path("./Data/input_images")
    data_path = Path("./Data/mixed_wiki")
    input_image_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)
    return OPENAI_API_KEY, input_image_path, data_path

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/download_images", methods=["GET"])
def download_images():
    """Download images using gdown from Google Drive."""
    input_image_path = Path("./Data/input_images")
    files = {
        "long_range_spec.png": "1nUhsBRiSWxcVQv8t8Cvvro8HJZ88LCzj",
        "model_y.png": "19pLwx0nVqsop7lo0ubUSYTzQfMtKJJtJ",
        "performance_spec.png": "1utu3iD9XEgR5Sb7PrbtMf1qw8T1WdNmF",
        "price.png": "1dpUakWMqaXR4Jjn1kHuZfB0pAXvjn2-i",
        "real_wheel_spec.png": "1qNeT201QAesnAP5va1ty0Ky5Q_jKkguV"
    }
    for filename, file_id in files.items():
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = input_image_path / filename
        try:
            gdown.download(url, str(output_path), quiet=False)
        except Exception as e:
            return jsonify({"status": "Error", "message": f"Error downloading {filename}: {e}"})
    return jsonify({"status": "Success", "message": "Images downloaded successfully"})

@app.route("/plot_images", methods=["GET"])
def plot_images():
    """Plot downloaded images."""
    input_image_path = Path("./Data/input_images")
    image_paths = [input_image_path / img_path for img_path in os.listdir(input_image_path)]
    plt.figure(figsize=(16, 9))
    for i, img_path in enumerate(image_paths):
        if img_path.exists():
            image = Image.open(img_path)
            plt.subplot(2, 3, i + 1)
            plt.imshow(image)
            plt.axis('off')
        if i >= 8:
            break
    plt.savefig('static/images/plot.png')
    return send_file('static/images/plot.png', mimetype='image/png')

@app.route("/download_additional_file", methods=["GET"])
def download_additional_file():
    """Download additional data using requests."""
    data_path = Path("./Data/mixed_wiki")
    url = "https://www.dropbox.com/scl/fi/mlaymdy1ni1ovyeykhhuk/tesla_2021_10k.htm?rlkey=qf9k4zn0ejrbm716j0gg7r802&dl=1"
    output_file = data_path / "tesla_2021_10k.htm"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
        return jsonify({"status": "Success", "message": "Additional data downloaded successfully!"})
    except Exception as e:
        return jsonify({"status": "Error", "message": f"Error downloading additional data: {e}"})

@app.route("/fetch_wikipedia_images", methods=["GET"])
def fetch_wikipedia_images():
    """Fetch images from Wikipedia for a given title."""
    title = request.args.get("title", "Tesla")
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url|dimensions|mime",
            "generator": "images",
            "gimlimit": "50",
        },
    ).json()
    image_urls = [
        page["imageinfo"][0]["url"]
        for page in response.get("query", {}).get("pages", {}).values()
        if page.get("imageinfo") and page["imageinfo"][0]["url"].endswith((".jpg", ".png"))
    ]
    return jsonify({"image_urls": image_urls})

def initialize_model():
    """Initialize the OpenAI multimodal LLM and storage context."""
    OPENAI_API_KEY, input_image_path, data_path = setup_environment()
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o", api_key=OPENAI_API_KEY, max_new_tokens=1500
    )

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED

    client = qdrant_client.QdrantClient(path=os.path.abspath("./Vector-Store/qdrant_mm_db"))
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )

    documents = SimpleDirectoryReader(data_path).load_data()

    try:
        index = MultiModalVectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        return index
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@app.route("/query", methods=["POST"])
def query():
    """Query the multimodal index and retrieve relevant images."""
    data = request.json
    query_text = data.get("query", "")
    retriever_engine = initialize_model().as_retriever(similarity_top_k=3, image_similarity_top_k=3)
    
    retrieval_results = retriever_engine.retrieve(query_text[:50])
    retrieved_images = [
        res_node.node.metadata["file_path"]
        for res_node in retrieval_results
        if isinstance(res_node.node, ImageNode)
    ]
    
    return jsonify({"retrieved_images": retrieved_images})

if __name__ == "__main__":
    app.run(debug=True)
