import streamlit as st
import pandas as pd
import pymongo
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dateutil.parser import parse
import numpy as np

# Set page configuration
st.set_page_config(page_title="Hotel Data Dashboard", layout="wide", initial_sidebar_state="expanded")

# Connect to MongoDB
@st.cache_resource
def connect_to_mongodb():
    try:
        client = pymongo.MongoClient("mongodb+srv://divyagangotri03:iYMKfEmQftNCpo8e@cluster0.wqwmf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        db = client["hotel_guests"]  # Replace with your actual database name
        return db
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

db = connect_to_mongodb()

# Function to load data from MongoDB collections
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_booking_data():
    bookings = list(db["new_bookings"].find({}, {'_id': 0}))
    df = pd.DataFrame(bookings)
    
    # Convert date columns to datetime
    df['check_in_date'] = pd.to_datetime(df['check_in_date'], errors='coerce')
    df['check_out_date'] = pd.to_datetime(df['check_out_date'], errors='coerce')
    return df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_dining_data():
    dining = list(db["dining_info"].find({}, {'_id': 0}))
    df = pd.DataFrame(dining)
    
    # Convert date columns to datetime
    df['check_in_date'] = pd.to_datetime(df['check_in_date'], errors='coerce')
    df['check_out_date'] = pd.to_datetime(df['check_out_date'], errors='coerce')
    df['order_time'] = pd.to_datetime(df['order_time'], errors='coerce')
    return df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_reviews_data():
    reviews = list(db["reviews_data"].find({}, {'_id': 0}))
    df = pd.DataFrame(reviews)
    
    # Convert date columns to datetime
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    return df

# Function to apply date filter to dataframes
def filter_by_date(df, start_date, end_date, date_column='check_in_date'):
    if date_column in df.columns:
        return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
    return df

# Sidebar for navigation and global filters
st.sidebar.title("Hotel Data Dashboard")

# Dashboard selection
dashboard_selection = st.sidebar.radio(
    "Select Dashboard",
    ["Bookings Dashboard", "Dining Dashboard", "Reviews Dashboard"]
)

# Global date filter
min_date = datetime(2023, 1, 1)
max_date = datetime(2025, 12, 31)

start_date = st.sidebar.date_input("From Date", min_date)
end_date = st.sidebar.date_input("To Date", max_date)

start_date = datetime.combine(start_date, datetime.min.time())
end_date = datetime.combine(end_date, datetime.max.time())

# Main content
if dashboard_selection == "Bookings Dashboard":
    st.title("Bookings Dashboard")
    
    try:
        # Load booking data
        bookings_df = load_booking_data()
        
        # Apply date filter
        filtered_df = filter_by_date(bookings_df, start_date, end_date)
        
        # Cuisine filter (multi-select)
        if 'Preferred_Cusine' in filtered_df.columns:
            cuisine_col = 'Preferred_Cusine'
        else:
            cuisine_col = 'Preferred Cusine'  # Alternative column name
            
        all_cuisines = filtered_df[cuisine_col].unique().tolist() if cuisine_col in filtered_df.columns else []
        selected_cuisines = st.multiselect("Select Cuisines", all_cuisines, default=all_cuisines)
        
        if selected_cuisines:
            filtered_df = filtered_df[filtered_df[cuisine_col].isin(selected_cuisines)]
        
        # Display filtered data count
        st.subheader(f"Total Bookings: {len(filtered_df)}")
        
        # Split layout into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Booking trend over time
            if not filtered_df.empty:
                booking_trend = filtered_df.groupby(filtered_df['check_in_date'].dt.strftime('%Y-%m')).size().reset_index(name='count')
                booking_trend.columns = ['Month', 'Number of Bookings']
                
                fig1 = px.line(booking_trend, x='Month', y='Number of Bookings', title='Booking Trend')
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Cuisine distribution
            if not filtered_df.empty:
                cuisine_counts = filtered_df[cuisine_col].value_counts().reset_index()
                cuisine_counts.columns = ['Cuisine', 'Count']
                
                fig2 = px.pie(cuisine_counts, values='Count', names='Cuisine', title='Preferred Cuisine Distribution')
                st.plotly_chart(fig2, use_container_width=True)
        
        # Guest distribution by age
        if not filtered_df.empty and 'age' in filtered_df.columns:
            st.subheader("Guest Age Distribution")
            fig3 = px.histogram(filtered_df, x='age', nbins=10, title='Guest Age Distribution')
            st.plotly_chart(fig3, use_container_width=True)
        
        # Stay duration analysis
        if not filtered_df.empty:
            filtered_df['stay_duration'] = (filtered_df['check_out_date'] - filtered_df['check_in_date']).dt.days
            
            st.subheader("Stay Duration Analysis")
            fig4 = px.box(filtered_df, x=cuisine_col, y='stay_duration', title='Stay Duration by Cuisine Preference')
            st.plotly_chart(fig4, use_container_width=True)
        
        # Raw data table
        with st.expander("View Raw Booking Data"):
            st.dataframe(filtered_df)
            
    except Exception as e:
        st.error(f"Error loading booking data: {e}")

elif dashboard_selection == "Dining Dashboard":
    st.title("Dining Dashboard")
    
    try:
        # Load dining data
        dining_df = load_dining_data()
        
        # Apply date filter (using order_time for dining)
        filtered_df = filter_by_date(dining_df, start_date, end_date, 'order_time')
        
        # Display filtered data count
        st.subheader(f"Total Dining Transactions: {len(filtered_df)}")
        
        # Split layout into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Popular dishes
            if not filtered_df.empty and 'dish' in filtered_df.columns:
                dish_counts = filtered_df['dish'].value_counts().reset_index()
                dish_counts.columns = ['Dish', 'Count']
                
                fig1 = px.bar(dish_counts.head(10), x='Dish', y='Count', title='Most Popular Dishes')
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Revenue by dish
            if not filtered_df.empty and 'dish' in filtered_df.columns and 'price_for_1' in filtered_df.columns and 'Qty' in filtered_df.columns:
                filtered_df['total_price'] = filtered_df['price_for_1'] * filtered_df['Qty']
                revenue_by_dish = filtered_df.groupby('dish')['total_price'].sum().reset_index()
                revenue_by_dish = revenue_by_dish.sort_values('total_price', ascending=False).head(10)
                
                fig2 = px.bar(revenue_by_dish, x='dish', y='total_price', title='Revenue by Dish')
                st.plotly_chart(fig2, use_container_width=True)
        
        # Order time analysis
        if not filtered_df.empty and 'order_time' in filtered_df.columns:
            filtered_df['hour'] = filtered_df['order_time'].dt.hour
            
            hour_counts = filtered_df.groupby('hour').size().reset_index(name='count')
            
            st.subheader("Order Time Distribution")
            fig3 = px.line(hour_counts, x='hour', y='count', title='Orders by Hour of Day')
            fig3.update_xaxes(tickmode='linear', tick0=0, dtick=1)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Cuisine preference analysis
        if not filtered_df.empty:
            if 'Preferred_Cusine' in filtered_df.columns:
                cuisine_col = 'Preferred_Cusine'
            else:
                cuisine_col = 'Preferred Cusine'  # Alternative column name
                
            if cuisine_col in filtered_df.columns and 'dish' in filtered_df.columns:
                cuisine_dish = filtered_df.groupby([cuisine_col, 'dish']).size().reset_index(name='count')
                cuisine_dish = cuisine_dish.sort_values('count', ascending=False).head(15)
                
                st.subheader("Cuisine Preferences and Dish Selection")
                fig4 = px.bar(cuisine_dish, x='dish', y='count', color=cuisine_col, barmode='group',
                            title='Dish Popularity by Cuisine Preference')
                st.plotly_chart(fig4, use_container_width=True)
        
        # Raw data table
        with st.expander("View Raw Dining Data"):
            st.dataframe(filtered_df)
            
    except Exception as e:
        st.error(f"Error loading dining data: {e}")

else:  # Reviews Dashboard
    st.title("Reviews Dashboard")
    
    try:
        # Load reviews data
        reviews_df = load_reviews_data()
        
        # Apply date filter
        filtered_df = filter_by_date(reviews_df, start_date, end_date, 'review_date')
        
        # Rating filter slider
        if 'Rating' in filtered_df.columns:
            min_rating, max_rating = st.slider(
                "Filter by Rating",
                min_value=1.0,
                max_value=10.0,
                value=(1.0, 10.0),
                step=0.1
            )
            
            filtered_df = filtered_df[(filtered_df['Rating'] >= min_rating) & (filtered_df['Rating'] <= max_rating)]
        
        # Display filtered data count
        st.subheader(f"Total Reviews: {len(filtered_df)}")
        
        # Split layout into two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            if not filtered_df.empty and 'Rating' in filtered_df.columns:
                fig1 = px.histogram(filtered_df, x='Rating', nbins=20, title='Rating Distribution')
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Rating trend over time
            if not filtered_df.empty and 'Rating' in filtered_df.columns and 'review_date' in filtered_df.columns:
                rating_trend = filtered_df.groupby(filtered_df['review_date'].dt.strftime('%Y-%m')).agg({'Rating': 'mean'}).reset_index()
                rating_trend.columns = ['Month', 'Average Rating']
                
                fig2 = px.line(rating_trend, x='Month', y='Average Rating', title='Rating Trend Over Time')
                fig2.update_yaxes(range=[0, 10])
                st.plotly_chart(fig2, use_container_width=True)
        
        # Sentiment analysis based on ratings
        if not filtered_df.empty and 'Rating' in filtered_df.columns:
            st.subheader("Rating Category Analysis")
            
            # Add rating category
            filtered_df['Rating_Category'] = pd.cut(
                filtered_df['Rating'],
                bins=[0, 3, 7, 10],
                labels=['Poor (0-3)', 'Average (4-7)', 'Excellent (8-10)']
            )
            
            rating_cat_counts = filtered_df['Rating_Category'].value_counts().reset_index()
            rating_cat_counts.columns = ['Category', 'Count']
            
            fig3 = px.pie(rating_cat_counts, values='Count', names='Category', title='Rating Categories')
            st.plotly_chart(fig3, use_container_width=True)
        
        # Raw data table with truncated review text
        with st.expander("View Raw Review Data"):
            display_df = filtered_df.copy()
            if 'Review' in display_df.columns:
                display_df['Review'] = display_df['Review'].str[:100] + '...'
            st.dataframe(display_df)
        
        # Individual review analysis
        if not filtered_df.empty and 'Review' in filtered_df.columns and 'Rating' in filtered_df.columns:
            st.subheader("Individual Review Analysis")
            
            # Sample or sort reviews for detailed viewing
            sorted_reviews = filtered_df.sort_values('Rating', ascending=False)
            
            # Show a few top and bottom reviews
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Highest Rated Reviews")
                for i, row in sorted_reviews.head(3).iterrows():
                    with st.container():
                        st.markdown(f"**Rating: {row['Rating']}/10**")
                        st.markdown(f"*Date: {row['review_date'].strftime('%Y-%m-%d') if pd.notnull(row['review_date']) else 'N/A'}*")
                        st.markdown(f"{row['Review'][:300]}...")
                        st.divider()
            
            with col4:
                st.subheader("Lowest Rated Reviews")
                for i, row in sorted_reviews.tail(3).iterrows():
                    with st.container():
                        st.markdown(f"**Rating: {row['Rating']}/10**")
                        st.markdown(f"*Date: {row['review_date'].strftime('%Y-%m-%d') if pd.notnull(row['review_date']) else 'N/A'}*")
                        st.markdown(f"{row['Review'][:300]}...")
                        st.divider()
            
    except Exception as e:
        st.error(f"Error loading reviews data: {e}")

# Footer
st.markdown("---")
st.markdown("Hotel Data Dashboard © 2025")