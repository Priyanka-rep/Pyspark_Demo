import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pyspark.sql.functions as f
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

spark = SparkSession.builder.appName("new").master("local[*]").enableHiveSupport().getOrCreate()
sc = spark.sparkContext

logger = logging.getLogger('py4j')

log4jLogger = spark._jvm.org.apache.log4j

LOGGER = log4jLogger.LogManager.getLogger(__name__)

LOGGER.info(f"Pyspark script logger initialized from {__name__}")


logger.info(" ** Reading input datasets ** ")

linkedin_users = spark.read.parquet("input_data/linkedin_users").persist()

user_x_connections = spark.read.parquet("input_data/user_x_connections").persist()

users_company_details = spark.read.parquet("input_data/users_company_details").persist()

# 1.	List the users who have at least one connection

ans1 = linkedin_users.join(user_x_connections, on="UserID", how="inner").\
    select(linkedin_users.UserID).distinct()

# 2.	List the most common employer among all users

ans2 = users_company_details.groupBy("CompanyName").agg(countDistinct("UserID").alias("cnt")).\
    orderBy(f.col("cnt").desc()).drop("cnt").limit(1)

# 3.    For each user list the oldest connection age and connected user

window = Window.partitionBy("UserID").orderBy("Connection_Datetime")

ans3 = user_x_connections.withColumn("rnk", f.row_number().over(window)).filter("rnk == 1").\
        withColumnRenamed("Connection_User_ID", "Oldest_Connection").\
        withColumn("Connection_Age",
                   concat_ws(" ", round(months_between(current_timestamp(), to_timestamp(f.col("Connection_Datetime"),
                                                                                   "dd-MMM-yy HH:mm:ss")),0).cast("int")
                             ,
                             f.lit("months")))\
        .drop("rnk")


# 4.	List employer and all the users in single column (comma separated).

user_x_company = users_company_details.join(linkedin_users, on="UserID", how="inner").\
    select(users_company_details.UserID, "CompanyName", "StartDate", "EndDate", "Name").persist()

ans4 = user_x_company.\
    select("CompanyName", "Name").\
    groupBy("CompanyName").agg(concat_ws(",", f.collect_list("Name")).alias("Names"))

# 5.	List currently who is the oldest employee of Intuit?

window1 = Window.partitionBy("CompanyName").orderBy("StartDate")

ans5 = user_x_company.filter(f.col('CompanyName') == 'Intuit').\
            withColumn("rnk", f.dense_rank().over(window1)).filter("rnk == 1").select("Name")

logger.info(" ** Users who have at least one connection ** ")
ans1.show()

logger.info(" ** Most common employer among all users ** ")
ans2.show()

logger.info(" ** For each user list the oldest connection age and connected user ** ")
ans3.show()

logger.info(" ** Employer and all the users in single column (comma separated) ** ")
ans4.show()

logger.info(" ** Oldest employee of Intuit ** ")
ans5.show()

input("Press enter to exit")

spark.stop()