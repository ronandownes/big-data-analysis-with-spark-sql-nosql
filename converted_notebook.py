#!/usr/bin/env python
# coding: utf-8

# # Stock Tweet and Price Analysis
# 
# In this notebook, we will analyze stock prices and tweets related to various companies. We will perform data preprocessing, exploratory data analysis (EDA), and implement time series forecasting models.
# 
# 
# ## Exploratory Data Analysis
# 
# ### Tweets and Financial Price Data
# The tweets and financial price data were obtained from the Twitter API and Yahoo Finance. These datasets were then stored and analyzed as per the project requirements.
# 
# ### Context
# 
# #### Data in "stocktweet.csv"
# - **Data Collection Period:** January 2020 - December 2020
# - **Number of Tweets:** 10,000
# - **Fields:**
#   - **ids:** The ID of the tweet (e.g., 100001)
#   - **date:** The date of the tweet (e.g., 01/01/2020)
#   - **ticker:** The ticker value for the company (e.g., AMZN)
#   - **tweet:** The text of the tweet (e.g., $AMZN Dow futures up by 100 points already)
# 
# #### Data in "stockprice" Folder
# - **Data Collection Period:** January 2020 - December 2020
# - **Companies (38 Tickers):**
#   - 'AAPL', 'ABNB', 'AMT', 'AMZN', 'BA', 'BABA', 'BAC', 'BKNG', 'BRK.A', 'BRK.B', 'CCL', 'CVX', 'DIS', 'FB', 'GOOG', 'GOOGL', 'HD', 'JNJ', 'JPM', 'KO', 'LOW', 'MA', 'MCD', 'MSFT', 'NFLX', 'NKE', 'NVDA', 'PFE', 'PG', 'PYPL', 'SBUX', 'TM', 'TSLA', 'TSM', 'UNH', 'UPS', 'V', 'WMT', 'XOM'
# - **Fields in Each CSV File:**
#   - **Date:** The date of the stock price (e.g., 01/01/2020)
#   - **Open:** The opening value of the stock price that day (e.g., 123.33)
#   - **High:** The highest value of the stock price that day (e.g., 125.45)
#   - **Low:** The lowest value of the stock price that day (e.g., 121.54)
#   - **Close:** The closing value of the stock price that day (e.g., 122.49)
#   - **Adj Close:** The adjusted closing value of the stock price that day (e.g., 122.49)
#   - **Volume:** The number of stocks traded that day (e.g., 100805600)
# 

# In[2]:


import zipfile
import pandas as pd
import os
import matplotlib.pyplot as plt

# Path to the zip file
zip_file_path = 'stock-tweet-and-price.zip'
extract_path = 'stock-tweet-and-price/'

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Verify the extracted files
extracted_files = os.listdir(extract_path)
print("Files in the extracted directory:", extracted_files)

# Analyse nested directory
nested_extract_path = os.path.join(extract_path, 'stock-tweet-and-price')
nested_files = os.listdir(nested_extract_path)
print("Files in the nested extracted directory:", nested_files)


# data/stock-tweet-and-price/stockprice


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the tweet dataset
tweets_df = pd.read_csv('stock-tweet-and-price/stocktweet/stocktweet.csv')
tweets_df.head()


# In[4]:


tweets_df.info()


# In[5]:


# Count the number of tweets for each ticker
ticker_counts = tweets_df['ticker'].value_counts()

# Get the number of unique companies (tickers) and total entries
num_companies = ticker_counts.nunique()
total_entries = len(tweets_df)

# Display the number of companies and total entries
print(f"Number of unique companies: {num_companies}")
print(f"Total entries: {total_entries}")

# Create a countplot bar chart for the number of tweets per ticker
plt.figure(figsize=(9, 6))
ax = sns.countplot(data=tweets_df, x='ticker', order=ticker_counts.index[:10])  # Display top 15 tickers
plt.title('Number of Tweets per Ticker (Top 10)')
plt.xlabel('Ticker')
plt.ylabel('Count of Tweets')
plt.xticks(rotation=45)

# Add values on the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Save the plot as a PNG file
plt.savefig('images/tweet_per_ticker.png')

# Show the plot
plt.show()


# In[6]:


# %whos


# In[7]:


# Count the number of tweets for each ticker and sort them in descending order
ticker_counts = tweets_df['ticker'].value_counts()

# Get the unique tickers ordered by frequency
ordered_unique_tickers = ticker_counts.index.tolist()

# Display the ordered unique tickers
print("Unique tickers in the dataset, ordered by tweet frequency:")
print(ordered_unique_tickers)


# From an inferential statistics perspective, choosing the top five companies like TSLA, AAPL, BA, DIS, and AMZN is beneficial because larger sample sizes reduce sampling variability and allow for more precise estimates 

# In[9]:


# Choose top 5 companies because  larger sample sizes 
# reduces sampling variability and allow for more precise estimates 

companies = ['TSLA', 'AAPL', 'BA', 'DIS', 'AMZN']
stock_data = {}

for company in companies:
    stock_data[company] = pd.read_csv(f'stock-tweet-and-price/stockprice/{company}.csv')


# Convert tweet dates to datetime and handle any invalid date formats
tweets_df['date'] = pd.to_datetime(tweets_df['date'], format='%d/%m/%Y', errors='coerce')

# Convert stock prices 'Date' columns to datetime and handle invalid formats
for company in companies:
    stock_data[company]['Date'] = pd.to_datetime(stock_data[company]['Date'], errors='coerce')


# In[10]:


# # Display the first 5 rows of tweets for each company
# for company in companies:
#     # Filter tweets related to the current company
#     company_tweets = tweets_df[tweets_df['ticker'] == company]
#     # Display the first 5 rows of the filtered tweets
#     print(f"First 5 tweets for {company}:")
#     display(company_tweets.head(5))


# ## Sentiment Analysis
# 
# In this step, we perform sentiment analysis on the tweets using the `TextBlob` library. Each tweet is assigned a sentiment polarity score ranging from -1 (negative) to 1 (positive). We then aggregate these scores by date to understand the overall sentiment trends for the stock-related tweets. This daily sentiment score will later be aligned with the stock price data for further analysis.
# 

# In[12]:


## Sentiment Analysis
from textblob import TextBlob

# Apply sentiment analysis
tweets_df['sentiment'] = tweets_df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)

# Aggregate sentiment by date
daily_sentiment = tweets_df.groupby('date')['sentiment'].mean().reset_index()

# Display the first few rows of the aggregated sentiment data
daily_sentiment.head()


# In[13]:


daily_sentiment.info()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define a color dictionary for the companies
color_dict = {
    'TSLA': 'black',
    'AAPL': 'red',
    'BA': 'blue',
    'DIS': 'green',
    'AMZN': 'darkgrey',
    'Aggregated': 'lightgrey'
}

company_colors = {
    'TSLA': 'black',
    'AAPL': 'red',
    'BA': 'blue',
    'DIS': 'green',
    'AMZN': 'darkgrey',
}

# Create a 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Histogram of Sentiment Scores with KDE', fontsize=16)

# Overall sentiment distribution (aggregated)
sns.histplot(data=tweets_df, x='sentiment', bins=30, kde=True, color=color_dict['Aggregated'], ax=axes[0, 0])
axes[0, 0].set_title('Aggregated Sentiment')
axes[0, 0].set_xlabel('Sentiment Score')
axes[0, 0].set_ylabel('Frequency')

# # Individual sentiment distributions for each company
# companies = ['TSLA', 'AAPL', 'BA', 'DIS', 'AMZN']
for i, company in enumerate(companies):
    row = (i + 1) // 3  # Calculate row index
    col = (i + 1) % 3   # Calculate column index
    company_tweets = tweets_df[tweets_df['ticker'] == company]
    
    sns.histplot(data=company_tweets, x='sentiment', bins=30, kde=True, color=color_dict[company], ax=axes[row, col])
    axes[row, col].set_title(f'{company} Sentiment')
    axes[row, col].set_xlabel('Sentiment Score')
    axes[row, col].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as a PNG file
plt.savefig('images/sentiment_histogram_with_kde_comparison.png')

# Show the plot
plt.show()


# In[15]:


# Create a 2x3 subplot grid
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Box Plot of Sentiment Scores', fontsize=16)

# Overall sentiment distribution (aggregated)
sns.boxplot(data=tweets_df, x='sentiment', color=color_dict['Aggregated'], ax=axes[0, 0])
axes[0, 0].set_title('Aggregated Sentiment')
axes[0, 0].set_xlabel('Sentiment Score')

# # Individual sentiment distributions for each company
# companies = ['TSLA', 'AAPL', 'BA', 'DIS', 'AMZN']
for i, company in enumerate(companies):
    row = (i + 1) // 3  # Calculate row index
    col = (i + 1) % 3   # Calculate column index
    company_tweets = tweets_df[tweets_df['ticker'] == company]
    
    sns.boxplot(data=company_tweets, x='sentiment', color=color_dict[company], ax=axes[row, col])
    axes[row, col].set_title(f'{company} Sentiment')
    axes[row, col].set_xlabel('Sentiment Score')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot as a PNG file
plt.savefig('images/sentiment_boxplot_comparison.png')

# Show the plot
plt.show()



# In[16]:


# Display the column names
tweets_df.columns


# In[17]:


import os

# Path to the directory containing the stock price CSV files
stockprice_dir = 'stock-tweet-and-price/stockprice'

# Check if all ticker names have corresponding CSV files in the directory
missing_files = []
for company in companies:
    company_csv_path = os.path.join(stockprice_dir, f'{company}.csv')
    if not os.path.exists(company_csv_path):
        missing_files.append(company)

if missing_files:
    print(f"Missing stock price data for tickers: {missing_files}")
else:
    print("All tickers have corresponding stock price CSV files.")


# In[18]:


import os
import pandas as pd

# Path to the directory containing the stock price CSV files
stockprice_dir = 'stock-tweet-and-price/stockprice'

# # List of companies to check
# companies = ['TSLA', 'AAPL', 'BA', 'DIS', 'AMZN']

# Check if all ticker names have corresponding CSV files in the directory
missing_files = []
files_with_missing_values = {}

for company in companies:
    company_csv_path = os.path.join(stockprice_dir, f'{company}.csv')
    
    # Check if the file exists
    if not os.path.exists(company_csv_path):
        missing_files.append(company)
    else:
        # Load the CSV and check for NaN or missing values
        df = pd.read_csv(company_csv_path)
        if df.isnull().values.any():
            files_with_missing_values[company] = df.isnull().sum().sum()  # Count total missing values

# Output the results
if missing_files:
    print(f"Missing stock price data for tickers: {missing_files}")
else:
    print("All tickers have corresponding stock price CSV files.")

if files_with_missing_values:
    print("\nThe following files have missing values:")
    for company, missing_count in files_with_missing_values.items():
        print(f"{company}.csv: {missing_count} missing values")
else:
    print("\nNo missing values found in any of the stock price CSV files.")


# In[19]:


# Ensure date columns are in the correct datetime format
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'], format='%Y-%m-%d', errors='coerce')

for company in companies:
    # Convert 'Date' column in stock data to datetime if it's not already
    stock_data[company]['Date'] = pd.to_datetime(stock_data[company]['Date'], format='%Y-%m-%d', errors='coerce')

# Merge sentiment data with stock price data for each selected company
merged_data = {}

for company in companies:
    # Merge the stock data with the daily sentiment data on the date
    merged_data[company] = pd.merge(stock_data[company], daily_sentiment, left_on='Date', right_on='date', how='left')
    
    # Drop the redundant 'date' column from the merged DataFrame
    merged_data[company].drop(columns=['date'], inplace=True)

# Display the first few rows of the merged data for one company (e.g., TSLA)
print("First few rows of the merged data for TSLA:")
print(merged_data['TSLA'].head())


# In[20]:


# Analyze the number of NaN values in the merged data for each company
for company in companies:
    num_missing = merged_data[company]['sentiment'].isna().sum()
    print(f"Number of NaN values in the 'sentiment' column for {company}: {num_missing}")


# In[21]:


# Ensure date columns are in the correct datetime format
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'], format='%Y-%m-%d', errors='coerce')

for company in companies:
    # Convert 'Date' column in stock data to datetime if it's not already
    stock_data[company]['Date'] = pd.to_datetime(stock_data[company]['Date'], format='%Y-%m-%d', errors='coerce')

# Merge sentiment data with stock price data for each selected company
merged_data = {}

for company in companies:
    # Merge the stock data with the daily sentiment data on the date
    merged_data[company] = pd.merge(stock_data[company], daily_sentiment, left_on='Date', right_on='date', how='left')
    
    # Drop the redundant 'date' column from the merged DataFrame
    merged_data[company].drop(columns=['date'], inplace=True)
    
    # Remove rows with NaN values
    merged_data[company].dropna(inplace=True)

# Display the first few rows of the merged data for one company (e.g., TSLA)
print("First few rows of the merged data for TSLA after removing NaN values:")
print(merged_data['TSLA'].head())


# In[22]:


# Merge sentiment data with stock price data for each selected company
merged_data = {}

for company in companies:
    # Merge the stock data with the daily sentiment data on the date
    merged_data[company] = pd.merge(stock_data[company], daily_sentiment, left_on='Date', right_on='date', how='left')

# Display the first few rows of the merged data for one company (e.g., AAPL)
merged_data['TSLA'].head()


# In[23]:


# Analyze the date range in the daily sentiment data
sentiment_date_range = (daily_sentiment['date'].min(), daily_sentiment['date'].max())
print(f"Sentiment data date range: {sentiment_date_range}")

# Analyze the date ranges in the stock price data for each company
for company in companies:
    stock_date_range = (stock_data[company]['Date'].min(), stock_data[company]['Date'].max())
    print(f"{company} stock data date range: {stock_date_range}")

    # Count NaN values in the merged data for each company
    nan_count = merged_data[company].isna().sum()
    print(f"NaN counts in merged data for {company}:\n{nan_count}\n")


# In[24]:


# Remove rows with NaN values from the merged data for each company
for company in companies:
    merged_data[company].dropna(inplace=True)

# Display the first few rows of the cleaned merged data for one company (e.g., TSLA)
print("First few rows of the cleaned merged data for TSLA:")
print(merged_data['TSLA'].head())

# Verify if there are any NaN values left in the merged data
for company in companies:
    nan_count = merged_data[company].isna().sum().sum()  # Sum of all NaN values in the dataframe
    print(f"Number of NaN values in {company} merged data: {nan_count}")


# In[25]:


# Merge sentiment data with stock price data for each selected company and drop NaN values
for company in companies:
    df = stock_data[company]
    df = df.merge(daily_sentiment, left_on='Date', right_on='date', how='left')
    df.drop(columns=['date'], inplace=True)
    df.dropna(inplace=True)  # Drop rows with NaN values
    stock_data[company] = df

# Display the first few rows of the merged data for one company (e.g., AAPL)
print(stock_data['AAPL'].head())


# ### Organizing Processed Data
# 
# This step involves creating a directory named `processed_data` to store the merged data for the selected companies. The new directory structure enhances data organization, making it easier to access and manage the processed files in subsequent stages of analysis. The code moves the files from their original location to this directory and lists the contents to verify the move.
# 

# In[27]:


import os

# Create a directory for storing the processed data CSVs if it doesn't already exist
output_directory = "processed_data"
os.makedirs(output_directory, exist_ok=True)

# Save merged data to CSV files
for company in companies:
    file_name = f"{company}_merged_data.csv"
    stock_data[company].to_csv(file_name, index=False)  # Save merged data to a CSV file

    # Move the newly created file to the processed_data directory
    new_path = os.path.join(output_directory, file_name)
    
    # Check if the file exists in the destination and remove it if it does
    if os.path.exists(new_path):
        os.remove(new_path)
    
    os.rename(file_name, new_path)

# List the files in the new directory to confirm they were moved
print(os.listdir(output_directory))



# ## Time Series Forecast
# 
# The project involves making a time series forecast of the CLOSE price for at least 5 companies using both the tweet data and financial price data. Forecasts are made for 1 day, 3 days, and 7 days into the future and displayed on a dynamic dashboard.
# 
# ### Project Requirements and Elements
# 
# - **Distributed Data Processing:** 
#   - The project incorporates a distributed data processing environment like Spark for part of the analysis.
# 
# - **Data Storage in SQL/NoSQL Databases:** 
#   - Source datasets are stored in SQL/NoSQL databases prior to processing using MapReduce or Spark (HBase, HIVE, Spark SQL, Cassandra, MongoDB).
#   - Data is loaded into the NoSQL database using an appropriate tool (Hadoop or Spark).
# 
# - **Post Map-Reduce Processing:** 
#   - Post MapReduce, the datasets are stored in an appropriate NoSQL database.
#   - The processed data is then extracted from the NoSQL database into another format (e.g., CSV) for further analysis in Python.
# 
# - **Comparative Analysis of Databases:** 
#   - A test strategy is devised to perform a comparative analysis of the capabilities of two databases (e.g., MySQL, MongoDB, Cassandra, HBase, CouchDB).
#   - Metrics are recorded, and a quantitative analysis is performed to compare the performance of the chosen database systems.
# 
# - **Sentiment Extraction Techniques:** 
#   - Evidence and justification of the sentiment extraction techniques used in the analysis.
# 
# - **Time-Series Forecasting Methods:** 
#   - At least two methods of time-series forecasting are explored, including:
#     - **1 Neural Network Model:** (e.g., LSTM)
#     - **1 Autoregressive Model:** (e.g., ARIMA, SARIMA)
#   - Since this is a short time series, considerations are made on how to handle the forecasting effectively.
# 
# - **Final Analysis and Justification:** 
#   - Justifications for the choices made in the final analysis are provided, along with the forecasts for 1 day, 3 days, and 7 days going forward.
# 
# - **Dynamic and Interactive Dashboard:** 
#   - The dashboard must be dynamic and interactive.
#   - The design rationale must express Tuft's principles.
# 

# ### Creating Lag Features for Stock Price and Sentiment
# This code creates lag features for the stock prices and sentiment scores for 1, 3, and 7 days. These features capture the temporal dependencies in the data, which are crucial for time-series forecasting. After creating these features, rows with missing values are removed.
# 

# In[30]:


# Function to create lag features for the target column
def create_lag_features(df, lags, target_col):
    """
    Creates lagged features for a specified column in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    lags (list): A list of integers indicating the number of lags to create.
    target_col (str): The column for which lag features are to be created.

    Returns:
    pd.DataFrame: The dataframe with new lagged features.
    """
    for lag in lags:
        # Creating lag features for the specified column
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Define the lag intervals to create
lags = [1, 3, 7]

# Loop through each company to create lag features for 'Close' and 'sentiment' columns
for company in companies:
    df = stock_data[company]
    # Creating lag features for the 'Close' price
    df = create_lag_features(df, lags, 'Close')
    # Creating lag features for the 'sentiment' scores
    df = create_lag_features(df, lags, 'sentiment')
    
    # Drop rows with NaN values introduced by the lagging process
    df.dropna(inplace=True)
    
    # Store the updated dataframe back to the stock_data dictionary
    stock_data[company] = df


# ### Visualization of Stock Prices
# In this step, we plot the historical stock prices for each selected company (Apple, Amazon, Google, Microsoft, and Tesla). Each stock price plot is saved as a PNG file in the `images` directory. This visualization provides a clear view of the stock price trends over time and serves as a reference for further analysis.
# 

# In[32]:


import matplotlib.pyplot as plt
import os

# Ensure the images directory exists
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)

# Plot stock price for each company 
for company in companies:
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data[company]['Date'], stock_data[company]['Close'], label=f'{company} Close Price', color=company_colors[company])
    plt.title(f'{company} Stock Price over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    image_path = os.path.join(images_dir, f'{company}_stock_price.png')
    plt.savefig(image_path)
    plt.show()


# ## Storage and Processing of Big Data Using Advanced Data Analytics Techniques
# 
# ### Big Data
# 
# - **Data Storage Preparation and Processing:** 
#   - Data storage is prepared and processed in a MapReduce/Spark environment to handle the large-scale data efficiently.
# 
# - **Comparative Analysis for Databases:**
#   - A comparative analysis of two databases, SQL and NoSQL, is performed using YCSB (Yahoo Cloud Serving Benchmark).
#   - The analysis includes metrics to evaluate the performance and capabilities of each database system.
# 
# - **Rationale for Data Processing and Storage Choices:** 
#   - The rationale behind the choice of data processing, storage methods, and programming languages is provided. This includes the justification for using Spark for distributed processing and the selection of SQL/NoSQL databases.
# 
# - **Architecture Design for Big Data Processing:** 
#   - The architecture for processing big data is designed to integrate necessary technologies, including HADOOP/SPARK, NoSQL/SQL databases, and programming tools.
#   - A diagram illustrating the design is presented in the report, accompanied by a detailed discussion.
# 
# - **MapReduce-Style Processing:** 
#   - In this context, MapReduce-style processing includes platforms such as Apache Spark, which facilitates efficient distributed data processing.
# 
# ### Advanced Data Analytics
# 
# - **Rationale, Evaluation, and Justification:**
#   - A detailed rationale and evaluation of the choices made during Exploratory Data Analysis (EDA), data wrangling, and the implementation of machine learning models and algorithms.
# 
# - **Hyperparameter Tuning Techniques:** 
#   - Evaluatio
# 

# ## Data Storage in SQL and NoSQL Databases
# In this step, we store the processed data into SQL (MySQL) and NoSQL (MongoDB) databases. The stock price data is stored in a MySQL database on our SQL VM, while the tweet data and sentiment scores are stored in a MongoDB database on our NoSQL VM. This approach allows us to efficiently query and manage both structured and unstructured data for further analysis and processing.
# 

# In[36]:


from sqlalchemy import create_engine

try:
    # Use the IP address of your VM
    mysql_engine = create_engine('mysql+pymysql://ronan:Zebra103!@192.168.0.190/stock_data_db')
    connection = mysql_engine.connect()
    
    # Export each company's DataFrame to a table in the MySQL database
    for company in companies:
        table_name = f"{company.lower()}_stock_data"
        stock_data[company].to_sql(table_name, mysql_engine, if_exists='replace', index=False)
        print(f"Data for {company} exported to MySQL table '{table_name}' successfully.")
    
    # Close the connection
    connection.close()
except Exception as e:
    print(f"MySQL export error: {e}")


# ## Setting Up MongoDB and Exporting Data
# 
# In this section, we set up MongoDB on the NoSQL virtual machine and configure it for remote access. We create a new user for secure database access and use the `pymongo` library to export processed tweet data to MongoDB.
# 
# ### Step 1: MongoDB Installation and Configuration on NoSQL VM
# - Installed MongoDB using `apt-get`.
# - Modified `/etc/mongodb.conf` to allow remote access.
# - Created a user `ronan` with password `Zebra103!` for secure database access.
# 
# ### Step 2: Connecting to MongoDB Using Python
# We use the `pymongo` library to connect to the MongoDB server at IP `192.168.0.219` and export the tweet data. The `MongoClient` is instantiated with the necessary credentials to ensure secure access.
# 
# ### Step 3: Data Export to MongoDB
# The tweet data is converted into a dictionary format and inserted into the MongoDB collection using the `insert_many()` method.
# 

# In[38]:


import pymongo

# Use the updated IP address of your VM
mongo_client = pymongo.MongoClient('mongodb://ronan:Zebra103!@192.168.0.190:27017/')
mongo_db = mongo_client['stock_sentiment_db']  # Database name
tweets_collection = mongo_db['tweets']  # Collection name

# Convert tweet DataFrame to dictionary records for MongoDB insertion
tweets_dict = tweets_df.to_dict('records')

# Insert the tweet data into the MongoDB collection
try:
    tweets_collection.insert_many(tweets_dict)
    print("Tweet data exported to MongoDB successfully.")
except Exception as e:
    print(f"MongoDB export error: {e}")


# In[41]:


from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("MySparkApp") \
    .config("spark.master", "spark://192.168.0.190:7077") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://192.168.0.190:9000") \
    .getOrCreate()

# # Example usage
# df = spark.read.csv("hdfs://192.168.0.190:9000/path/to/data.csv", header=True, inferSchema=True)  # Adjust the path as needed
# df.show()


# In[84]:


# Check columns in each company's dataframe
for company in companies:
    print(f"Columns in {company}'s dataframe: {stock_data[company].columns}")


# In[ ]:





# In[43]:


from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import matplotlib.pyplot as plt
import os




# Loop through each company and generate forecasts
for company in companies:
    df = stock_data[company]
    
    # Convert the 'Date' column to datetime and set it as the index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Use the 'Close' price for forecasting
    y = df['Close']

    # Train an ARIMA model (example with (p=5, d=1, q=0), adjust as needed)
    model = ARIMA(y, order=(5, 1, 0))
    model_fit = model.fit()

    # Forecast the next 1, 3, and 7 days
    forecast_1day = model_fit.forecast(steps=1)
    forecast_3days = model_fit.forecast(steps=3)
    forecast_7days = model_fit.forecast(steps=7)

    # Prepare actual and predicted data for plotting (example with the last 100 points)
    y_test = y[-100:]
    y_pred_1day = np.append(y_test.values[:-1], forecast_1day)
    y_pred_3days = np.append(y_test.values[:-3], forecast_3days)
    y_pred_7days = np.append(y_test.values[:-7], forecast_7days)

    # Plot and save 1-day forecast
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Prices', color=company_colors[company])
    plt.plot(y_test.index, y_pred_1day, label='1-Day Prediction', color='red')
    plt.title(f'{company} Stock Price Prediction - 1 Day')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    image_path = os.path.join(images_dir, f'{company}_forecast_1day.png')
    plt.savefig(image_path)
    plt.show()

    # Plot and save 3-day forecast
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Prices', color=company_colors[company])
    plt.plot(y_test.index, y_pred_3days, label='3-Day Prediction', color='orange')
    plt.title(f'{company} Stock Price Prediction - 3 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    image_path = os.path.join(images_dir, f'{company}_forecast_3days.png')
    plt.savefig(image_path)
    plt.show()

    # Plot and save 7-day forecast
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test, label='Actual Prices', color=company_colors[company])
    plt.plot(y_test.index, y_pred_7days, label='7-Day Prediction', color='green')
    plt.title(f'{company} Stock Price Prediction - 7 Days')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    image_path = os.path.join(images_dir, f'{company}_forecast_7days.png')
    plt.savefig(image_path)
    plt.show()


# In[92]:


import pymongo
import pandas as pd

# Connect to MongoDB
mongo_client = pymongo.MongoClient('mongodb://osboxes.org:Zebra103!@localhost:27017/')
mongo_db = mongo_client['stock_sentiment_db']  # Database name
tweets_collection = mongo_db['tweets']  # Collection name

# Load your Twitter data (assuming it's in a DataFrame called tweets_df)
tweets_df = pd.read_csv('stock-tweet-and-price/stocktweet/stocktweet.csv')

# Convert tweet DataFrame to dictionary records for MongoDB insertion
tweets_dict = tweets_df.to_dict('records')

# Insert the tweet data into the MongoDB collection
try:
    tweets_collection.insert_many(tweets_dict)
    print("Tweet data exported to MongoDB successfully.")
except Exception as e:
    print(f"MongoDB export error: {e}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




