import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
df = pd.read_csv('hotel_bookings.csv')

# Set page configuration
st.set_page_config(page_title='Hotel Booking Demand Analysis', layout='wide')

# Sidebar - Booking Cancellation Analysis
st.sidebar.header('Booking Cancellation Analysis')
selected_features = st.sidebar.multiselect('Select Features', ('lead_time', 'arrival_date_week_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies'))
y_feature = st.sidebar.selectbox('Select Target Feature', ('is_canceled',))

# Display the cancellation analysis plot
if selected_features and y_feature:
    selected_columns = selected_features + [y_feature]
    cancellation_df = df[selected_columns]
    cancellation_plot = sns.pairplot(cancellation_df, hue=y_feature)
    st.pyplot(cancellation_plot)

# Sidebar - Customer Segmentation
st.sidebar.header('Customer Segmentation')
cluster_features = st.sidebar.multiselect('Select Features for Clustering', ('lead_time', 'arrival_date_week_number', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies'))

# Display the customer segmentation plot
if cluster_features:
    cluster_df = df[cluster_features]
    st.subheader('Customer Segmentation')
    st.dataframe(cluster_df.head())

# Main content - EDA
st.title('Hotel Booking Demand Analysis')
st.subheader('Exploratory Data Analysis (EDA)')

# Distribution of canceled and non-canceled bookings
st.subheader('Canceled vs. Non-canceled Bookings')
canceled_count = df['is_canceled'].value_counts()
st.bar_chart(canceled_count)

# Average lead time for bookings
st.subheader('Average Lead Time for Canceled vs. Non-canceled Bookings')
lead_time_mean = df.groupby('is_canceled')['lead_time'].mean()
st.bar_chart(lead_time_mean)

# Guest distribution by market segment
st.subheader('Guest Distribution by Market Segment')
segment_count = df['market_segment'].value_counts()
st.bar_chart(segment_count)

# Average daily rate by room type
st.subheader('Average Daily Rate by Room Type')
room_type_mean = df.groupby('reserved_room_type')['adr'].mean()
st.bar_chart(room_type_mean)

# Correlation heatmap
st.subheader('Correlation Heatmap')
corr_matrix = df.corr(numeric_only=True)  # Fix for the FutureWarning

# Enhance the heatmap visibility
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
st.pyplot(plt)

# Seasonal Booking Trends
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + df['arrival_date_month'] + '-01')
seasonal_bookings = df.groupby(df['arrival_date'].dt.to_period('M'))['is_canceled'].value_counts().unstack().fillna(0)
seasonal_bookings.rename(columns={0: 'Not Canceled', 1: 'Canceled'}, inplace=True)

st.subheader('Seasonal Booking Trends')
st.line_chart(seasonal_bookings, use_container_width=True)

# Guest Nationality Analysis
top_nationalities = df['country'].value_counts().head(10)
st.subheader('Top Guest Nationalities')
st.bar_chart(top_nationalities)

# Length of Stay Analysis
df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
stay_length = df.groupby('total_stay_nights')['is_canceled'].mean()
st.subheader('Average Cancellation Rate by Length of Stay')
st.bar_chart(stay_length)

# Repeat Guests Analysis
repeat_guests = df['previous_cancellations'].apply(lambda x: 'Repeat' if x > 0 else 'New')
repeat_guests_count = repeat_guests.value_counts()
st.subheader('Repeat vs. New Guests')
st.bar_chart(repeat_guests_count)

# Customer Demographics and Preferences
st.subheader('Customer Demographics and Preferences')
customer_prefs = ['country', 'market_segment', 'reserved_room_type', 'meal']
selected_prefs = st.multiselect('Select Customer Preferences', customer_prefs, default=customer_prefs)
if selected_prefs:
    prefs_df = df[selected_prefs]
    st.dataframe(prefs_df.head())

# Cancellation Reasons
# Let's assume you have a separate column 'cancellation_reason' that categorizes the reasons for cancellations.
# You can analyze and visualize the distribution of cancellation reasons.

# Channel Distribution
channel_distribution = df['distribution_channel'].value_counts()
st.subheader('Booking Channel Distribution')
st.bar_chart(channel_distribution)

# Booking Lead Time and Cancellations
lead_time_vs_cancelation = df.groupby('lead_time')['is_canceled'].mean()
st.subheader('Cancellation Rate vs. Booking Lead Time')
st.line_chart(lead_time_vs_cancelation)

# Occupancy Rates Analysis
df['occupied'] = df['is_canceled'].apply(lambda x: 'Occupied' if x == 0 else 'Canceled')
occupancy_rates = df.groupby('arrival_date')['occupied'].value_counts().unstack().fillna(0)
st.subheader('Occupancy Rates Over Time')
st.area_chart(occupancy_rates, use_container_width=True)

# Top Guest Nationalities - World Heat Map
world_map_data = df['country'].value_counts().reset_index()
world_map_data.columns = ['country', 'count']
fig = px.choropleth(world_map_data, locations='country', locationmode='country names', color='count', hover_name='country', 
                    color_continuous_scale='viridis', title='Top Guest Nationalities', projection='natural earth')
st.plotly_chart(fig, use_container_width=True)

# Average Daily Rate by Room Type - Box Plot
fig2 = px.box(df, x='reserved_room_type', y='adr', title='Average Daily Rate by Room Type',
              labels={'reserved_room_type': 'Room Type', 'adr': 'Average Daily Rate (USD)'})
st.plotly_chart(fig2, use_container_width=True)

# Footer
st.sidebar.markdown('**Note:** Adjust the options in the sidebar to explore different aspects of the data.')
