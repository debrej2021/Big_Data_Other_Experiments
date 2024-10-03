# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

# Initialize Spark session
spark = SparkSession.builder \
    .appName('PySparkExample') \
    .getOrCreate()

# Read data from a CSV file into a DataFrame
df = spark.read.csv('path/to/your/data.csv', header=True, inferSchema=True)

# Show the first few rows of the DataFrame
df.show(5)

# Select specific columns
df_selected = df.select('column1', 'column2')

# Filter rows where 'column1' is greater than a threshold
df_filtered = df_selected.filter(df_selected['column1'] > 50)

# Calculate the average of 'column2'
average_value = df_filtered.agg(avg('column2')).collect()[0][0]
print(f'Average of column2: {average_value}')

# Stop the Spark session
spark.stop()
