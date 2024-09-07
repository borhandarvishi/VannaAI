from openai import OpenAI

# openAI Embedding

api_key = "sk-proj"

client = OpenAI(
    api_key=api_key,
)

# text-embedding-3-small
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
    return embedding
