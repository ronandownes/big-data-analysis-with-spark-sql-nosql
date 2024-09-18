# big-data-analysis-with-spark-sql-nosql
# Big Data Analysis with Spark, SQL, and NoSQL

This project demonstrates big data processing and time series forecasting using Apache Spark, SQL, and NoSQL databases. It focuses on the analysis of financial stock price data and social media sentiment to predict stock trends. 

## Project Overview
The project involves:
- Setting up a big data processing environment using Ubuntu virtual machines (`database_vm` and `processing_vm`).
- Utilizing MySQL for structured data storage and MongoDB for unstructured tweet data.
- Performing distributed data processing using Apache Spark.
- Implementing time series forecasting to predict stock prices.
- Conducting comparative analysis of SQL and NoSQL database performance.

## Technologies Used
- **Apache Spark**: For distributed data processing.
- **MySQL**: For structured data management.
- **MongoDB**: For unstructured data storage.
- **Python (Jupyter Notebooks)**: For data analysis and visualization.
- **VirtualBox**: For creating virtual environments on Ubuntu OS.

## Setup Instructions
1. Clone the repository to your local machine.
2. Set up the virtual environments (`database_vm` and `processing_vm`) using the provided instructions in the repository.
3. Install required software:
   - Apache Spark
   - MySQL
   - MongoDB
   - Jupyter Notebook
4. Run the Jupyter notebooks for data analysis and forecasting.

## Project Structure
- `database_vm/`: Configuration files and scripts for the database virtual machine.
- `processing_vm/`: Configuration files and scripts for the processing virtual machine.
- `notebooks/`: Jupyter notebooks for data processing and analysis.
- `data/`: Sample data used in the project.

## Usage
- Use the Jupyter notebooks to perform data analysis, sentiment extraction, and time series forecasting.
- Compare the performance of SQL (MySQL) and NoSQL (MongoDB) databases using provided scripts.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
