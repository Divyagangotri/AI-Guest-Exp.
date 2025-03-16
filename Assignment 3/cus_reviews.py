import streamlit as st
import pandas as pd
import os
from langchain_together import TogetherEmbeddings
from pinecone import Pinecone
from together import Together
import smtplib
from email.message import EmailMessage
import datetime

# Set environment variables 
os.environ["TOGETHER_API_KEY"] = 'ceb0de08eea5f49206a26abce9e45968b48beeb6814b2059c88342377fe9e0f2'

# File Path
file_path = 'C:/Users/Neelam Ramesh/Desktop/Ass-2/reviews_data.xlsx'

# Initialize Pinecone
pc = Pinecone(api_key='pcsk_5R5bP6_75EKcHPQePsfoFVT6Bk9iRTo6UMeuDYcTnhKGAo8wr2LAAbJ27D5qmvamKfyY9L')
index = pc.Index(host="sample-movies1-pqsjbw9.svc.aped-4627-b74a.pinecone.io")

# Initialize Together Embedding Model
embeddings = TogetherEmbeddings(
    model='togethercomputer/m2-bert-80M-8k-retrieval',
    together_api_key=os.environ["TOGETHER_API_KEY"]
)

# Initialize Together client
client = Together(api_key=os.environ["TOGETHER_API_KEY"])

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Customer Feedback Form</h1>", unsafe_allow_html=True)

# Input Fields
customer_id = st.text_input("Customer ID")
review = st.text_area("Enter Your Review")
room_number = st.text_input("Room Number")
rating = st.slider("Rate Your Experience (1-10)", 1, 10, 5)
currently_staying = st.radio("Are you currently staying in the hotel?", ("Yes", "No"))

if st.button("Submit Review"):
    if not customer_id or not review or not room_number:
        st.warning("Please fill all the fields")
    else:
        # Read existing data
        try:
            df = pd.read_excel(file_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Customer_ID", "Review", "Room_Number", "Rating", "Currently_Staying", "Review_Date"])

        # Add new review
        review_date = datetime.datetime.now().strftime("%Y-%m-%d")
        new_data = pd.DataFrame([{
            "Customer_ID": customer_id,
            "Review": review,
            "Room_Number": room_number,
            "Rating": rating,
            "Currently_Staying": currently_staying,
            "Review_Date": review_date
        }])
        df = pd.concat([df, new_data], ignore_index=True)

        # Save to Excel
        df.to_excel(file_path, index=False)

        # Embed and store in Pinecone
        review_embedding = embeddings.embed_query(review)
        index.upsert(
            vectors=[
                {
                    "id": str(customer_id),
                    "values": review_embedding,
                    "metadata": {
                        "Customer_ID": customer_id,
                        "Review": review,
                        "Room_Number": room_number,
                        "Rating": rating,
                        "Currently_Staying": currently_staying,
                        "Review_Date": review_date
                    }
                }
            ]
        )

        st.success("Review submitted successfully and stored in the vector database!")

        # Send Email to Manager if Customer is Currently Staying
        if currently_staying == "Yes":
            try:
                sender_email = "divya.gangotri03@gmail.com"  # Replace with your email
                sender_password = "yacp xeph saij gkkv"  # Replace with your email password
                manager_email = "2021csm.r70@svec.edu.in"  # Replace with manager's email

                msg = EmailMessage()
                msg["Subject"] = "New Real-Time Review from a Staying Customer"
                msg["From"] = sender_email
                msg["To"] = manager_email
                msg.set_content(f"""
                Customer ID: {customer_id}
                Room Number: {room_number}
                Rating: {rating}/10
                Review: {review}
                """)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                    server.login(sender_email, sender_password)
                    server.send_message(msg)

                st.success("Manager notified of the real-time review.")
            except Exception as e:
                st.error(f"Error sending email: {e}")
