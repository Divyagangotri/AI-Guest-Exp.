import streamlit as st
import pandas as pd
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
from together import Together
import os

# Set environment variables 
os.environ["TOGETHER_API_KEY"] = 'ceb0de08eea5f49206a26abce9e45968b48beeb6814b2059c88342377fe9e0f2'

# Load data
df = pd.read_excel('C:/Users/Neelam Ramesh/Desktop/Ass-2/reviews_data.xlsx')

# Initialize Pinecone
pc = Pinecone(api_key='pcsk_5R5bP6_75EKcHPQePsfoFVT6Bk9iRTo6UMeuDYcTnhKGAo8wr2LAAbJ27D5qmvamKfyY9L')
index = pc.Index(host="sample-movies1-pqsjbw9.svc.aped-4627-b74a.pinecone.io")

# Initialize Together embedding model with proper API key
embeddings = TogetherEmbeddings(
    model='togethercomputer/m2-bert-80M-8k-retrieval',
    together_api_key=os.environ["TOGETHER_API_KEY"]
)

# Initialize Together client
client = Together(api_key=os.environ["TOGETHER_API_KEY"])

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Hotel Customers' Sentiments Analysis</h1>", unsafe_allow_html=True)

query = st.text_input("Enter a query about customer reviews:", "How is the food quality?")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
rating_filter = st.slider("Select Rating Filter", 1, 10, (1, 10))

if st.button("Analyze Sentiments"):
    query_embedding = embeddings.embed_query(query)
    
    start_date_str = int(start_date.strftime("%Y%m%d"))
    end_date_str = int(end_date.strftime("%Y%m%d"))
    
    results = index.query(
        vector=query_embedding,
        top_k=5,
        namespace="",
        filter={
            "review_date": {"$gte": start_date_str, "$lte": end_date_str},
            "Rating": {"$gte": rating_filter[0], "$lte": rating_filter[1]}
        },
        include_metadata=True
    )
    
    matches = results["matches"]
    
    if not matches:
        st.warning("No matching reviews found")
    else:
        # Extract review IDs from metadata or ID
        matched_ids = []
        for match in matches:
            if "metadata" in match and "review_id" in match["metadata"]:
                matched_ids.append(int(match["metadata"]["review_id"]))
            elif "id" in match:
                # Fallback to using the ID directly if it's the review ID
                try:
                    matched_ids.append(int(match["id"]))
                except ValueError:
                    pass
        
        # Proceed only if we found matching IDs
        if matched_ids:
            req_df = df[df["review_id"].isin(matched_ids)]
            
            if not req_df.empty:
                concatenated_reviews = " ".join(req_df["Review"].tolist())
                
                response = client.chat.completions.create(
                    model="meta-llama/Llama-Vision-Free",
                    messages=[{
                        "role": "user", 
                        "content": f"""
                        Briefly summarize the overall sentiment of customers based on these reviews - 
                        {concatenated_reviews} and query of the manager {query}.
                        Stick to specific query of manager, and keep it concise.
                        Do not mention the name of the hotel.
                        """}]
                )
                
                st.subheader("Sentiment Analysis")
                st.write(response.choices[0].message.content)

                st.subheader("Matching Reviews")
                st.dataframe(req_df[["Review", "Rating"]])
            else:
                st.warning("No matching reviews found in the dataset")
        else:
            st.warning("Could not extract review IDs from the matches")