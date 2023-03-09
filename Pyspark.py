import sys

from pyspark.sql import SparkSession

# from pyspark.sql.functions import avg, sum
# from pyspark.sql.window import Window


def main():
    # Setup Spark
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    appName = "PySpark"
    master = "local"
    # Create Spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()
    # Window Function for calculating the batting average
    # rolling_window =" Window.partitionBy('batter').orderBy('days_diff').rangeBetween(-100, -1)"
    # rolling_sums = (
    # sum('Hit').over(rolling_window).alias('rolling_hits'),
    # sum('atBat').over(rolling_window).alias('rolling_atbats')
    # )
    # rolling_avg = avg('rolling_hits' / 'rolling_atbats').over(rolling_window).alias('rolling_avg')
    sql = "select * from batter_counts"

    database = "baseball"
    user = "admin"
    password = "1234"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df.show()


if __name__ == "__main__":
    sys.exit(main())
