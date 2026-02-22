from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model_name="text-embedding-ada-002",
    dimensions=32,
)

documents = [
    "Lahore is the capital of Pakistan",
    "Islamabad is the capital of Pakistan",
]

result = embedding.embed_documents(documents)
print(str(result))