from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=300,
)

documents = [
    "Lahore is the capital of Pakistan",
    "Islamabad is the capital of Pakistan",
]

query="What is the capital of Pakistan?"

doc_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key = lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity Score is: ", score)