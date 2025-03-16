# Import necessary libraries
import os
import pandas as pd
import numpy as np
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone, ServerlessSpec
from together import Together

# Set API keys
if not os.getenv("TOGETHER_API_KEY"):
    os.environ["TOGETHER_API_KEY"] = "ceb0de08eea5f49206a26abce9e45968b48beeb6814b2059c88342377fe9e0f2"

# Load and prepare data
def load_data(file_path):
    """Load hotel review data from Excel file"""
    df = pd.read_excel(file_path)
    return df

# Generate embeddings
def generate_embeddings(reviews, batch_size=128):
    """Generate embeddings for review text in batches"""
    # Initialize the TogetherEmbeddings model
    embeddings = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval"
    )
    
    # Process embeddings in batches
    embedding_list = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        embedding_list.extend(batch_embeddings)
        print(f"Processed {i + len(batch)} / {len(reviews)} reviews")
    
    return embeddings, embedding_list

# Prepare metadata for vector database
def prepare_metadata(df):
    """Create metadata dictionary for each review"""
    metadata_list = df.apply(lambda row: {
        "customer_id": int(row["customer_id"]),
        "review_date": row["review_date_numeric"],
        "Rating": int(row["Rating"]),
        "review_id": row['review_id']
    }, axis=1).tolist()
    
    return metadata_list

# Initialize and setup Pinecone vector database
def setup_pinecone(api_key, index_name, dimension):
    """Initialize Pinecone and create index if needed"""
    pc = Pinecone(api_key='pcsk_5R5bP6_75EKcHPQePsfoFVT6Bk9iRTo6UMeuDYcTnhKGAo8wr2LAAbJ27D5qmvamKfyY9L')
    
    # Create a new index (will skip if already exists)
    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            deletion_protection='enabled',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new index: sample-movies1")
    except Exception as e:
        print(f"Index may already exist or another error occurred: {e}")
    
    # Connect to index
    index = pc.Index(host="sample-movies1-pqsjbw9.svc.aped-4627-b74a.pinecone.io")
    return index

# Upload vectors to Pinecone
def upload_to_pinecone(index, embedding_list, metadata_list, batch_size=100):
    """Upload vectors to Pinecone in batches"""
    for i in range(0, len(embedding_list), batch_size):
        batch_vectors = [
            (str(i + j), embedding_list[i + j], metadata_list[i + j])
            for j in range(min(batch_size, len(embedding_list) - i))
        ]
        index.upsert(vectors=batch_vectors)
        print(f"Upserted batch from {i} to {i + len(batch_vectors)}")

# Query the vector database
def query_pinecone(index, embeddings, query_text, top_k=5, filter_params=None):
    """Query Pinecone for similar reviews"""
    query_embedding = embeddings.embed_query(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace="",
        include_metadata=True,
        filter=filter_params
    )
    
    return results

# Analyze sentiment using LLM
def analyze_sentiment(reviews_text):
    """Use LLM to analyze sentiment of reviews"""
    client = Together()
    response = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[{
            "role": "user", 
            "content": f"Briefly Summarize the overall sentiment of customers about food and restaurant based on these reviews - {reviews_text}. Don't mention the name of the hotel"
        }]
    )
    return response.choices[0].message.content

# Main function to orchestrate the workflow
def main():
    # Configuration
    file_path = "C:/Users/Neelam Ramesh/Desktop/Ass-2/reviews_data.xlsx"

    pinecone_api_key = 'pcsk_5R5bP6_75EKcHPQePsfoFVT6Bk9iRTo6UMeuDYcTnhKGAo8wr2LAAbJ27D5qmvamKfyY9'
    index_name = 'sample-movies1'
    dimension = 768
    
    # 1. Load data
    print("Loading review data...")
    df = load_data(file_path)
    reviews = df["Review"].tolist()
    
    # 2. Generate embeddings
    print("Generating embeddings...")
    embeddings_model, embedding_list = generate_embeddings(reviews)
    
    # 3. Prepare metadata
    print("Preparing metadata...")
    metadata_list = prepare_metadata(df)
    
    # 4. Set up Pinecone
    print("Setting up Pinecone...")
    index = setup_pinecone(pinecone_api_key, index_name, dimension)
    
    # 5. Upload vectors to Pinecone
    print("Uploading vectors to Pinecone...")
    upload_to_pinecone(index, embedding_list, metadata_list)
    
    # 6. Query for food-related reviews
    print("Querying for food-related reviews...")
    filter_params = {
        "Rating": {"$lte": 9},
        "review_date": {"$gte": 20240101, "$lte": 20240108}
    }
    results = query_pinecone(
        index, 
        embeddings_model, 
        "What are some of the reviews that mention restaurant, food, lunch, breakfast, dinner",
        filter_params=filter_params
    )
    
    # 7. Extract matching reviews
    matches = results["matches"]
    matched_ids = [int(match["metadata"]["review_id"]) for match in matches]
    req_df = df[df["review_id"].isin(matched_ids)]
    concatenated_reviews = " ".join(req_df["Review"].tolist())
    
    # 8. Analyze sentiment using LLM
    print("Analyzing sentiment...")
    sentiment_summary = analyze_sentiment(concatenated_reviews)
    print("\nSentiment Analysis Result:")
    print(sentiment_summary)
    
if __name__ == "__main__":
    main()