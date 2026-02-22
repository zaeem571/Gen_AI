from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Use the correct class for Hugging Face Remote execution
# 2. Use a proper embedding model (MiniLM is standard and fast)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
)

# 3. Generate the embedding
text = "Lahore is the capital of Pakistan"
result = embeddings.embed_query(text)

# Print the first 5 numbers just to prove it works (the full list is huge)
print(f"Embedding generated! Vector length: {len(result)}")
print(f"First 5 values: {result[:5]}")