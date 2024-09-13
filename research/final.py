import os
import requests
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv
import warnings
from pathlib import Path
import gdown
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.core import PromptTemplate
import qdrant_client
import ssl
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Payload indexes have no effect in the local Qdrant.")

# Load environment variables from a .env file
load_dotenv()

# Get your OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    logging.error("OPENAI API key not found. Set it in the environment variables or in a .env file.")
    raise ValueError("OPENAI API key not found. Set it in the environment variables or in a .env file.")

# Create necessary directories
input_image_path = Path("../Data/input_images")
data_path = Path("../Data/mixed_wiki")
input_image_path.mkdir(parents=True, exist_ok=True)
data_path.mkdir(parents=True, exist_ok=True)

logging.info("Environment prepared successfully!")

# Define the URLs and output paths
files = {
    "long_range_spec.png": "1nUhsBRiSWxcVQv8t8Cvvro8HJZ88LCzj",
    "model_y.png": "19pLwx0nVqsop7lo0ubUSYTzQfMtKJJtJ",
    "performance_spec.png": "1utu3iD9XEgR5Sb7PrbtMf1qw8T1WdNmF",
    "price.png": "1dpUakWMqaXR4Jjn1kHuZfB0pAXvjn2-i",
    "real_wheel_spec.png": "1qNeT201QAesnAP5va1ty0Ky5Q_jKkguV"
}

# Download each file
for filename, file_id in files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = input_image_path / filename
    try:
        gdown.download(url, str(output_path), quiet=False)
        logging.info(f"Downloaded {filename} successfully.")
    except Exception as e:
        logging.error(f"Error downloading {filename}: {e}")

def plot_images(image_paths):
    plt.figure(figsize=(16, 9))
    for i, img_path in enumerate(image_paths):
        if img_path.exists():
            try:
                image = Image.open(img_path)
                plt.subplot(2, 3, i + 1)
                plt.imshow(image)
                plt.axis('off')
            except Exception as e:
                logging.error(f"Error opening image {img_path}: {e}")
        if i >= 8:
            break
    plt.show()

# Gather image paths
image_paths = [input_image_path / img_path for img_path in os.listdir(input_image_path)]
plot_images(image_paths)

# Load image documents
try:
    image_documents = SimpleDirectoryReader(input_image_path).load_data()
    logging.info("Image documents loaded successfully.")
except Exception as e:
    logging.error(f"Error loading image documents: {e}")
    raise

# Initialize OpenAI multimodal LLM
try:
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4o", api_key=OPENAI_API_KEY, max_new_tokens=1500
    )
    logging.info("OpenAI multimodal LLM initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing OpenAI multimodal LLM: {e}")
    raise

# Generate descriptions
try:
    response_1 = openai_mm_llm.complete(
        prompt="Generate detailed text description for each image.",
        image_documents=image_documents,
    )
    logging.info("Descriptions generated successfully.")
    print(response_1)
except Exception as e:
    logging.error(f"Error generating descriptions: {e}")

def get_wikipedia_images(title):
    try:
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
        return image_urls
    except Exception as e:
        logging.error(f"Error fetching Wikipedia images for {title}: {e}")
        return []

# List of Wikipedia titles to fetch
wiki_titles = [
    "Tesla Model Y", "Tesla Model X", "Tesla Model 3", "Tesla Model S",
    "Kia EV6", "BMW i3", "Audi e-tron", "Ford Mustang",
    "Porsche Taycan", "Rivian", "Polestar",
]

# Fetch text and images
image_uuid = 0
MAX_IMAGES_PER_WIKI = 5

for title in wiki_titles:
    try:
        # Fetch text
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page.get("extract", "")

        with open(data_path / f"{title}.txt", "w") as fp:
            fp.write(wiki_text)

        # Fetch images
        images_per_wiki = 0
        list_img_urls = get_wikipedia_images(title)

        for url in list_img_urls:
            if url.endswith((".jpg", ".png", ".svg")):
                image_uuid += 1
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(data_path / f"{image_uuid}.jpg", "wb") as fp:
                            fp.write(response.content)
                        images_per_wiki += 1
                        if images_per_wiki >= MAX_IMAGES_PER_WIKI:
                            break
                except Exception as e:
                    logging.error(f"Error downloading image from URL {url}: {e}")
    except Exception as e:
        logging.error(f"Error processing Wikipedia title {title}: {e}")

logging.info("Data collection completed!")

# Configure SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Create Qdrant vector store
try:
    client = qdrant_client.QdrantClient(path=os.path.abspath("../Vector-Store/qdrant_mm_db"))
    text_store = QdrantVectorStore(client=client, collection_name="text_collection")
    image_store = QdrantVectorStore(client=client, collection_name="image_collection")
    storage_context = StorageContext.from_defaults(
        vector_store=text_store, image_store=image_store
    )
    logging.info("Qdrant vector store created successfully.")
except Exception as e:
    logging.error(f"Error creating Qdrant vector store: {e}")
    raise

# Build multimodal index
try:
    documents = SimpleDirectoryReader(data_path).load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    logging.info("Multimodal index built successfully!")
except Exception as e:
    logging.error(f"An error occurred while building the multimodal index: {e}")
    raise

MAX_TOKENS = 50
retriever_engine = index.as_retriever(
    similarity_top_k=3, image_similarity_top_k=3
)

def retrieve_and_display(query):
    try:
        retrieval_results = retriever_engine.retrieve(query[:MAX_TOKENS])
        retrieved_images = [
            res_node.node.metadata["file_path"]
            for res_node in retrieval_results
            if isinstance(res_node.node, ImageNode)
        ]

        if retrieved_images:
            plt.figure(figsize=(15, 5))
            for i, img_path in enumerate(retrieved_images):
                plt.subplot(1, len(retrieved_images), i + 1)
                img = Image.open(img_path)
                plt.imshow(img)
                plt.axis('off')
            plt.show()
    except Exception as e:
        logging.error(f"Error retrieving and displaying images for query '{query}': {e}")

# Example usage
retrieve_and_display("What is the best electric Sedan?")

qa_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)

query_engine = index.as_query_engine(
    llm=openai_mm_llm, text_qa_template=qa_tmpl
)

def multimodal_rag_query(query_str):
    try:
        response = query_engine.query(query_str)
        print("Answer:", str(response))

        print("\nSources:")
        for text_node in response.metadata.get("text_nodes", []):
            display_source_node(text_node, source_length=200)

        if response.metadata.get("image_nodes"):
            plt.figure(figsize=(15, 5))
            for i, img_node in enumerate(response.metadata["image_nodes"]):
                plt.subplot(1, len(response.metadata["image_nodes"]), i + 1)
                img = Image.open(img_node.metadata["file_path"])
                plt.imshow(img)
                plt.axis('off')
            plt.show()
    except Exception as e:
        logging.error(f"Error executing multimodal RAG query '{query_str}': {e}")

# Example usage
multimodal_rag_query("Compare the design features of Tesla Model S and Rivian R1")
