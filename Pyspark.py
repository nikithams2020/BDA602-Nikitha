# import sys
# Reference Blog for Pyspark and Mariadb connection
# https://kontext.tech/article/1061/pyspark-read-data-from-mariadb-database
# Lecture Slides for transformer settings
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import datediff, min
from pyspark.sql.window import Window

database = "baseball"
user = "admin"
password = "1234"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


class TransformerFunc(Transformer):
    @keyword_only
    def __init__(self):
        super(TransformerFunc, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(
        self, spark: SparkSession, df_bc: DataFrame, df_game: DataFrame
    ) -> DataFrame:
        # Join the DataFrames on game_id and select the required columns
        df = (
            df_bc.join(df_game, "game_id")
            .select("batter", "game_id", "local_date", "Hit", "atBat")
            .withColumn(
                "days_diff",
                datediff(
                    "local_date", min("local_date").over(Window.partitionBy("batter"))
                ),
            )
            .filter("atBat != 0")
        )

        # Register the DataFrame as a temporary view
        df.createOrReplaceTempView("temp_view")

        # Execute the SQL query and save the result as a DataFrame
        result_df = spark.sql(
            """
            SELECT
                bc.batter,
                bc.game_id,
                COALESCE((
                    SELECT
                        AVG(bc2.Hit * 1.0 / bc2.atBat)
                    FROM
                        temp_view bc2
                        JOIN temp_view g2 ON g2.game_id = bc2.game_id
                    WHERE
                        bc2.batter = bc.batter AND
                        bc2.days_diff - g2.days_diff BETWEEN 1 AND 100
                ), 0) AS batting_avg
            FROM
                temp_view bc
                JOIN temp_view g ON g.game_id = bc.game_id
            """
        )
        return result_df

    # bc2.game_id <> bc.game_id
    # AND
    # bc2.batter = bc.batter
    # AND
    # Show the result
    # result_df.show(10)

    # Setup Spark
    # spark = SparkSession.builder.master("local[*]").getOrCreate()

    # appName = "PySpark"
    # master = "local"
    # Create Spark session
    # spark = SparkSession.builder.appName(appName).master(master).getOrCreate()


def main():
    table_game = "select local_date, game_id from baseball.game"
    table_batter = "select batter, atBat, Hit, game_id from baseball.batter_counts"
    appName = "PySpark"
    master = "local"
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()
    # load the batter_counts table as a DataFrame
    df_bc = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .option("query", table_batter)
        .load()
    )
    df_game = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .option("query", table_game)
        .load()
    )

    obj = TransformerFunc()
    result = obj._transform(spark, df_bc, df_game)
    result.show()


if __name__ == "__main__":
    # sys.exit()
    main()
