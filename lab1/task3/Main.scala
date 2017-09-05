package se.kth.spark.lab1.task3

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    //val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val conf = new SparkConf().setAppName("lab2")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    //val filePath = "src/main/resources/millionsong.txt"
    //val filePath = "/home/saurabh/all.txt"
    val filePath = "hdfs://10.0.104.163:8020/Projects/SaurabhML/Resources/all.txt"
    val obsDF: DataFrame = sc.textFile(filePath).toDF()
    //obsDF.show(5)

    //Step1: tokenize each row

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("new_value")
      .setPattern("[\\d\\.-]+")
      .setGaps(false)
      //.setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    val tokenized = regexTokenizer.transform(obsDF)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
    arr2Vect.setInputCol("new_value").setOutputCol("vector_value")
    val vector = arr2Vect.transform(tokenized)

    //Step4: extract the label(year) into a new column

    val lSlicer = new VectorSlicer()
      .setInputCol("vector_value")
      .setOutputCol("year")
      .setIndices(Array(0))

    val year = lSlicer.transform(vector)

    //Step5: convert type of the label from vector to double (use our Vector2Double)

    val v2d = new Vector2DoubleUDF((vector) => vector(0))
      .setInputCol("year")
      .setOutputCol("year_double")
    val sth = v2d.transform(year)

    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF)

    val lShifter = new DoubleUDF(x => x - 1922)
      .setInputCol("year_double")
      .setOutputCol("label")
    val year3 = lShifter.transform(sth)

    //Step7: extract just the 3 first features in a new vector column

    val fSlicer = new VectorSlicer()
      .setInputCol("vector_value")
      .setOutputCol("features")
      .setIndices(Array(1,2,3,4,5,6,7,8,9,10,11,12))
    val dataset = fSlicer.transform(year3)

    // Linear Regression

    val myLR = new LinearRegression()
      .setMaxIter(45)
      .setRegParam(0.001)
      .setFeaturesCol("features")
      .setLabelCol("label")
      //.setElasticNetParam(0.9)


    //Step8: put everything together in a pipeline
    //val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR ))

    //Step9: generate model by fitting the rawDf into the pipeline
    //val pipelineModel = pipeline.fit(obsDF)

    //Step10: transform data with the model - do predictions
    //val transformed_data = pipelineModel.transform(obsDF)

    //val lrStage =
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR ))
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)  // training data
    val lrModelSummary = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel].summary

    println("The root mean squared error is: " + lrModelSummary.rootMeanSquaredError)  // RMSE

    //val predictions = pipelineModel.transform(obsDF)   // Making predictions on the Testing data
    //predictions.show(5)

    //do prediction - print first 5
    print(lrModelSummary.predictions.show(5))


    // Predictions
    //val prediction = pipelineModel.(obsDF)

    // Fit on the training data

    //print rmse of our model

  sc.stop()

  }
}