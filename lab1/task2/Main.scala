package se.kth.spark.lab1.task2

import se.kth.spark.lab1._
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SQLContext

import scala.tools.scalap.Main

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "/home/saurabh/all.txt"
    // val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF()
    //rawDF.show(5)

    //Step1: tokenize each row

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("new_value")
      .setPattern(",")
      //.setOutputCol("[\\d\\.-]+")


    //Step2: transform with tokenizer and show 5 rows
    val tokenized = regexTokenizer.transform(rawDF)
    //tokenized.show(5)
    //tokenized.select("new_value").take(5).foreach(println)


    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    arr2Vect.setInputCol("new_value").setOutputCol("vector_value")
    val vector = arr2Vect.transform(tokenized)
    //vector.show(5)

    //Step4: extract the label(year) into a new column

    val lSlicer = new VectorSlicer()
      .setInputCol("vector_value")
      .setOutputCol("year")
      .setIndices(Array(0))

    val year = lSlicer.transform(vector)
   // println(output.select("userFeatures", "features").first())

    year.show(5)


    //Step5: convert type of the label from vector to double (use our Vector2Double)

    val v2d = new Vector2DoubleUDF((vector) => vector(0))
      .setInputCol("year")
      .setOutputCol("year_double")


    val sth = v2d.transform(year)
    sth.show(5)


    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 

    val lShifter = new DoubleUDF(x => x - 1922)
      .setInputCol("year_double")
      .setOutputCol("label")

    val year3 = lShifter.transform(sth)
    year3.show(5)

    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer()
      .setInputCol("vector_value")
      .setOutputCol("features")
      .setIndices(Array(1,2,3))

    val dataset = fSlicer.transform(year3)
    dataset.show()

    //Linear regression
   // val lr = new LinearRegression()



    //Step8: put everything together in a pipeline
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect, lSlicer, v2d, lShifter, fSlicer ))

    //Step9: generate model by fitting the rawDf into the pipeline
    val pipelineModel = pipeline.fit(rawDF)


    //Step10: transform data with the model - do predictions
    val transformed_data = pipelineModel.transform(rawDF)
    transformed_data.show(5)

    //Step11: drop all columns from the dataframe other than label and features
    val new_dropped = transformed_data.drop("value","new_value","vector_value","year","year_double")
    new_dropped.show(5)
  }
}