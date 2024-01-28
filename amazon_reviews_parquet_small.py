from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import sys

args = sys.argv

# myDir = "/mnt/c/Users/xxxxx/Downloads/"
myDir = args[1]
myParquet = myDir + "/amazon_reviews_2015.snappy.parquet"

spark = SparkSession \
                .builder \
                .appName("myapp") \
                .master("local") \
                .config("spark.executor.memory", "8g") \
                .config("spark.sql.parquet.binaryAsString","true") \
                .getOrCreate()

conf = spark.sparkContext.getConf()
print("# spark.executor.memory = ", conf.get("spark.executor.memory"))
print("# spark.executor.memoryOverhead = ", conf.get("spark.executor.memoryOverhead"))

df=spark.read.parquet(myParquet).select("star_rating","review_id","review_body")

small_df = spark.createDataFrame(df.head(500000))
small_df.show()
small_df.write.mode("overwrite").parquet(myDir + "amazon_reviews_2015_small.snappy.parquet")
