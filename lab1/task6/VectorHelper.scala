package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}

/*
The interface of Vector is quite limiting, so a solution where you call the .toArray()
to get an array out of it and use functional combinators like map on this array and create
a new DenseVector as result will be accepted
*/
object VectorHelper {
  // Dot product of two vectors
  def dot(v1: Vector, v2: Vector): Double = {
    var x1:Array[Double] = v1.toArray
    var x2:Array[Double] = v2.toArray
    var dotpr:Double = 0.00
    for ( i <- 0 to x1.size - 1 ){
      dotpr += x1(i) * x2(i)
    }
    return dotpr
  }

  // Dot product of a vector and a scalar
  def dot(v: Vector, s: Double): Vector = {
    var xout:Array[Double] = v.toArray
    for( i <-0 to xout.size - 1 ){
      xout(i) = xout(i) * s
    }
    return Vectors.dense(xout)

  }

  // Sum - Implement addition of 2 vectors
  def sum(v1: Vector, v2: Vector): Vector = {
    val x1:Array[Double] = v1.toArray
    val x2:Array[Double] = v2.toArray
    for( i <- 0 to x1.size -1 ){
      x1(i) = x1(i) + x2(i)
    }
    return Vectors.dense(x1)
   }

  //fill - create a vector of predefined size and initialize it with the predefined value
  def fill(size: Int, fillVal: Double): Vector = {
    val arrayrep:Array[Double] = Array.fill(size)(fillVal)
    return Vectors.dense(arrayrep)


  }

}