package com.eightbit85

import cats.data.State
import cats.free.Free

package object monnn {

  type NeuralNetworkF[A] = Free[NeuralNetwork, A]

  case class Config(
                     alpha: Double,
                     threshold: Double,
                     epsilon: Double,
                     limit: Int,
                     features: List[TrainingPoint],
                     size: Int
                   )

  case class Result(thetas: List[ThetaLayer], cost: List[Double])
  case class TrainingPoint(o: List[List[Double]], y: List[Double])

  case class ThetaBucket(thetas: Array[Double], result: Array[Double], counts: Array[Int]) {
    def add(idx: Int, i: Double): Unit = {
      result(idx) += i * thetas(counts(idx))
      counts(idx) += 1
    }

    def pull(idx: Int): Double = {
      counts(idx) += 1
      result(idx) * thetas(counts(idx))
    }

    def diffTheta(idx: Int, prev: Double): Unit = {
      thetas(counts(idx)) += result(idx) * prev
      counts(idx) += 1
    }
  }

  object ThetaBucket {
    def create(thetas: List[Double], filterSize: Int): ThetaBucket = {
      val ar = thetas.toArray
      ThetaBucket(ar, Array.fill(filterSize)(ar(0)), Array.fill(filterSize)(1))
    }
  }

  implicit class RangeWithFirstError(rng: Range) {

    def firstError[A](p: Int => Either[A, Int]): Either[A, Int] = {
      val it = rng.iterator
      var amnt = 0
      while (it.hasNext) {
        val a = it.next()
        p(a) match {
          case Left(i) => return Left(i)
          case Right(i) => amnt += i
        }
      }
      Right(amnt)
    }

  }

}
