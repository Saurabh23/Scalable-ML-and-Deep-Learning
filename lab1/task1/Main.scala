package se.kth.spark.lab1.task1


import breeze.linalg.max
import se.kth.spark.lab1._
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //val rawDF = ???

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    // Delimiter = ',' , Number of Features = 13, Data Types = Double
    rdd.take(5).foreach(println)

    //Step2: split each row into an array of features
    val recordsRdd = rdd.map( row => row.split(",").map( x => x.toDouble))
    //recordsRdd.foreach(println)

    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(song => (song(0), song(1), song(2), song(3)))
    //val songs2 = songsRdd.flatMap(x =>x)
    //songs2.take(8).foreach(println)
    //songsRdd.foreach(println)

    //Step4: convert your rdd into a dataframe
    val songsDf = songsRdd.toDF()
    songsDf.show(8)

    // Q1. How many songs there are in the DataFrame?
    // Q3. What is the min, max and mean value of the year column?
    songsDf.describe().show()



    //Q2: How many songs were released between the years 1998 and 2000?
    print(songsDf.filter(songsDf("_1") > 1998 && songsDf("_1") < 2000).count)

    //Q4: Show the number of songs per year between the years 2000 and 2010?
    print(songsDf.filter(songsDf("_1") > 2000 && songsDf("_1") < 2010).groupBy("_1").count.show())



  }
}
