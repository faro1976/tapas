package it.utiu.tapas.stats.analyzer

import it.utiu.tapas.base.AbstractAnalyzerActor
import org.apache.spark.sql.SparkSession
import it.utiu.tapas.util.Consts

import org.apache.spark.sql.SQLContext
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.ml.linalg.Vector
import java.text.SimpleDateFormat
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.{ DataFrame, SparkSession }
import java.nio.file.Files
import java.nio.file.Paths
import java.io.File
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.Row
import scala.collection.JavaConverters._
import akka.actor.Props
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.feature.VectorAssembler
import java.nio.file.StandardOpenOption
import akka.actor.ActorRef
import org.apache.spark.sql.types.Decimal
import org.apache.spark.sql.types.DataTypes
import org.apache.spark.sql.types.DecimalType
import it.utiu.tapas.base.AbstractBaseActor

object BTCAnalyzerActor {
  def props(): Props = Props(new BTCAnalyzerActor())

}

class BTCAnalyzerActor() extends AbstractAnalyzerActor(Consts.CS_BTC) {
  override def doInternalAnalysis(spark: SparkSession): (Array[String], scala.collection.immutable.List[Row]) = {


    import spark.implicits._
    import org.apache.spark.sql.functions._

//    val df1 = spark.read.json(HDFS_CS_PATH + "*")    
//    val df2 = df1.select("context.cache.since", "data.transactions_24h", "data.difficulty", "data.volume_24h", "data.mempool_transactions", "data.mempool_size", "data.mempool_tps", "data.mempool_total_fee_usd", "data.average_transaction_fee_24h", "data.nodes", "data.inflation_usd_24h", "data.average_transaction_fee_usd_24h", "data.market_price_usd", "data.next_difficulty_estimate", "data.suggested_transaction_fee_per_byte_sat")
//
//    //do analytics
//    //instant values: difficulty, nodes, mempool_transactions, market_price_usd
//    //last 24h values: transactions_24h, volume_24h, average_transaction_fee_24h, inflation_usd_24h
//
//    val dfHourlyWindow = df2
//      .groupBy(window(df2.col("since"), "1 hour"))
//      .agg(avg("market_price_usd").as("hourly_average_price"))
//    dfHourlyWindow.show()
//    val df3_1 = df2.withColumn("next_hour", date_trunc("HOUR", col("since") + expr("INTERVAL 1 HOURS")))
//
//    val df3 = df3_1.join(dfHourlyWindow, col("next_hour") === date_trunc("HOUR", col("window.start")))
//      .withColumnRenamed("hourly_average_price", "next_avg_price").withColumnRenamed("window", "next_window")
//
//    //add yyyy-MM-dd HH date
//    val df4 = df3.withColumn("datehour_only", date_format(to_timestamp(col("since"), "yyyy-MM-dd HH:mm:ss"), "yyyy-MM-dd HH"))
//
//    //instant and last24h values, compute average
//    val df5 = df4.groupBy("datehour_only").agg(mean("difficulty").as("avgDifficulty"), mean("nodes").as("avgNodes"), mean("mempool_transactions").as("avgMempoolTxs"), mean("market_price_usd").as("avgPriceUSD"), mean("transactions_24h").as("avgTx24h"), mean("volume_24h").as("avgVolume24h"), mean("average_transaction_fee_24h").as("avgTxFee24h"), mean("inflation_usd_24h").as("avgInflatUSD24h"), mean("next_avg_price").as("avgNextAvgPrice"))
//    val df6 = df5.withColumn("avgDifficulty", col("avgDifficulty").cast(DecimalType(25, 4))).withColumn("avgNodes", col("avgNodes").cast(DecimalType(14, 4))).withColumn("avgMempoolTxs", col("avgMempoolTxs").cast(DecimalType(14, 4)))
//      .withColumn("avgPriceUSD", col("avgPriceUSD").cast(DecimalType(14, 4))).withColumn("avgTx24h", col("avgTx24h").cast(DecimalType(14, 4))).withColumn("avgVolume24h", col("avgVolume24h").cast(DecimalType(25, 4)))
//      .withColumn("avgTxFee24h", col("avgTxFee24h").cast(DecimalType(14, 4))).withColumn("avgInflatUSD24h", col("avgInflatUSD24h").cast(DecimalType(14, 4))).withColumn("avgNextAvgPrice", col("avgNextAvgPrice").cast(DecimalType(14, 4)))
//
//    df6.show()
//    val ret = df6.sort("datehour_only")


    val df1 = spark.read.format("csv").option("header", "true").option("inferSchema", "true").option("delimiter", "\t").load(AbstractBaseActor.HDFS_URL + "/" + Consts.CS_BTC + "-dump/*").withColumn("strTime", date_format(col("time"),"yyyy-MM-dd"))
    val df2 = df1.groupBy("strTime").agg(avg("fee_per_kb"), avg("output_total"), avg("size"), count("hash")).withColumn("avg(output_total)_abs", col("avg(output_total)").cast(DecimalType(14, 4)))
    val ret = df2.sort("strTime")
    ret.show()
    
    //compute correlation matrix 
    val assembler = new VectorAssembler().setInputCols(Array("size", "io_count", "output_total", "fee_per_kb", "fee")).setOutputCol("features").setHandleInvalid("skip")
    val df3 = assembler.transform(df1.withColumn("io_count", col("input_count")+col("output_count")).select("size", "io_count", "output_total", "fee_per_kb", "fee"))    
    df3.show()
    val Row(coeff1: Matrix) = Correlation.corr(df3, "features").head
    println(s"Pearson correlation matrix:\n $coeff1")

//    val Row(coeff2: Matrix) = Correlation.corr(df3, "features", "spearman").head
//    println(coeff2.toString(Int.MaxValue, Int.MaxValue))

    val buff = new StringBuilder("\n")
    for (v <- coeff1.colIter) {
        buff.append(v.toArray.mkString(",") + "\n")
    }
    writeFile(ANALYTICS_OUTPUT_FILE + ".corrMtx2.csv", buff.toString, None)

    
    (ret.columns, ret.collectAsList().asScala.toList)
  }
}