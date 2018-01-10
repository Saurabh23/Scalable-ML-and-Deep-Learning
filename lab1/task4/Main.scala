package se.kth.spark.lab1.task4


import org.apache.spark._
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    //Small Size dataset
    //val obsDF: DataFrame = ???

    val obsDF: DataFrame = sc.textFile(filePath).toDF()
    //obsDF.show(5)

    //Step1: tokenize each row

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("new_value")
      .setPattern(",")

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
    println(dataset)

    // Linear Regression

    val myLR = new LinearRegression()
      .setMaxIter(50)
      .setRegParam(0.9)
      .setElasticNetParam(0.1)
      .setFeaturesCol("features")
      .setLabelCol("label")

    //val lrStage =
    val pipeline = new Pipeline().setStages(Array(regexTokenizer,arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR ))
    //val myLR = ???
    //val lrStage = ???
    //val pipeline = ???
    //val cvModel: CrossValidatorModel =
    //val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel]

    //build the parameter grid by setting the values for maxIter and regParam
    val paramGrid = new ParamGridBuilder()
      .addGrid(myLR.maxIter, Array(5, 25, 45, 70, 100, 1000))
      .addGrid(myLR.regParam, Array(0.001, 0.1, 0.6, 1.5, 2.0 ))
      .addGrid(myLR.elasticNetParam, Array( 0.9))
      .build()


     val evaluator = new RegressionEvaluator()
     //create the cross validator and set estimator, evaluator, paramGrid
     val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    val cvModel = cv.fit(obsDF)
    cvModel.transform(obsDF)

    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel]
    val bestModelSummary =  bestModel.summary
    //val lrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(6).asInstanceOf[LinearRegressionModel].summary

    //print best model RMSE to compare to previous
    println("Root mean squared error is: " + bestModelSummary.rootMeanSquaredError)
    println("Regularisation Param is: " + bestModel.getRegParam)
    println("Max Iterations is: " + bestModel.getMaxIter)
    println("Elastic Net Param is: " + bestModel.getElasticNetParam)



    //print rmse of our model
    //do prediction - print first k
  }
}
