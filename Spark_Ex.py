# word_count_dataframe.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, lower, col

# Create Spark session
spark = SparkSession.builder \
    .appName("WordCountDataFrame") \
    .getOrCreate()

# Read the text file into a DataFrame
df = spark.read.text("lenear.docx")

# Split lines into words and explode into individual rows
words_df = df.select(explode(split(col("value"), " ")).alias("word"))

# Convert words to lowercase
words_df = words_df.select(lower(col("word")).alias("word"))

# Group by words and count
word_counts_df = words_df.groupBy("word").count()

# Show the results
word_counts_df.show()

# Stop Spark session
spark.stop()
