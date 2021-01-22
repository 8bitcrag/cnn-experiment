package com.eightbit85.monnn

import com.eightbit85.monnn.helpers.MatrixHelpers.{sigmoid, transpose}
import cats.instances.double._
import com.eightbit85.monnn.helpers.SeqHelper
import com.eightbit85.monnn.helpers.SeqHelpers.listSeqHelper

import scala.annotation.tailrec

sealed abstract class ThetaLayer {
  val thetas: List[List[Double]]
  val previousSize: Int

  def gradientPlaceHolders: Array[Array[Double]] = thetas.map(th => th.map(_ => 0.0).toArray).toArray
  def liftActivations(input: List[List[Double]]): List[List[Double]]
  def getNextDiffWithRespectToNodes(currentDiff: List[Double]): List[Double]
  def getGradients(currentDiff: List[Double], prevActs: List[Double], grads: Array[Array[Double]]): Array[Array[Double]]
}

case class Normal(thetas: List[List[Double]]) extends ThetaLayer {
  override val previousSize: Int = thetas.head.length - 1
  private val gnd = SeqHelper[List].mtxVecToProduct(transpose(thetas).tail)

  override def liftActivations(i: List[List[Double]]): List[List[Double]] = SeqHelper[List].toActivations(thetas)(i.head) :: i

  override def getNextDiffWithRespectToNodes(currentDiff: List[Double]): List[Double] = {
    gnd(currentDiff) // use cats Eval for this? to specify call-by-whatever for performance?
  }

  override def getGradients(currentDiff: List[Double], prevActs: List[Double], grads: Array[Array[Double]]): Array[Array[Double]] = {
    var row: Int = 0
    currentDiff.foreach(d => {
      var col: Int = 1
      grads(row)(0) += d
      prevActs.foreach(a => {
        grads(row)(col) += d*a
        col += 1
      })
      row += 1
    })
    grads
  }
}

case class Convolutional(thetas: List[List[Double]], maps: List[List[Int]], depth: Int, filterSize: Int) extends ThetaLayer {
  private val revThetas = thetas.reverse
  override val previousSize: Int = maps.length * depth
  private val arrayVersion: Array[Array[Double]] = thetas.map(_.toArray).toArray

  // Forward Prop
  def fpf(thetas: Array[Double], input: List[Double]): List[Double] = {
    forwardPropFilter(ThetaBucket(thetas, Array.fill(filterSize)(thetas(0)), Array.fill(filterSize)(1)), input)
  }

  @tailrec
  final def forwardPropChannel(submaps: List[List[Int]], tb: ThetaBucket, input: List[Double]): List[Double] = {
    submaps match {
      case Nil => input
      case ::(h, t) =>
        h.foreach(m => tb.add(m, input.head))
        forwardPropChannel(t, tb, input.tail)
    }
  }

  @tailrec
  final def forwardPropFilter(tb: ThetaBucket, input: List[Double]): List[Double] = {
    input match {
      case Nil => tb.result.map(sigmoid).toList
      case _ =>
        val channeled = forwardPropChannel(maps, tb, input)
        forwardPropFilter(tb, channeled)
    }
  }

  def forwardProp(input: List[Double]): List[Double] = {
    revThetas.map(th => ThetaBucket.create(th, filterSize))
      .map(tb => forwardPropFilter(tb, input))
      .reduceLeft((a, c) => c ::: a)
  }

  override def liftActivations(input: List[List[Double]]): List[List[Double]] = {
    forwardProp(input.head) :: input
  }

  // BackProp activations
  @tailrec
  private def backPropChannel(maps: List[List[Int]], thetas: ThetaBucket, result: Array[Double])(i: Int): Int = {
    maps match {
      case Nil => i
      case ::(h, t) =>
        h.foreach(j => result(i) += thetas.pull(j))
        backPropChannel(t, thetas, result)(i+1)
    }
  }

  private def backPropFilters(depth: Int, filterSize: Int, maps: List[List[Int]], thetas: Array[Array[Double]], result: Array[Double])(diffs: Array[Double]): List[Double] = {
    val ph = 0 until depth
    thetas.indices.foreach(i => {
      val tb = ThetaBucket(thetas(i), diffs.slice(i * filterSize, i * filterSize + filterSize), Array.fill(filterSize)(0))
      val b = backPropChannel(maps, tb, result)(_)
      Function.chain(ph.map(_ => b))(0)
    })
    result.toList
  }

  override def getNextDiffWithRespectToNodes(currentDiff: List[Double]): List[Double] = {
    backPropFilters(depth, filterSize, maps, arrayVersion, Array.fill(previousSize)(0.0))(currentDiff.toArray)
  }

  // BackProp Thetas
  @tailrec
  final def backPropThetaChannel(mp: List[List[Int]], tb: ThetaBucket, input: List[Double]): List[Double] = {
    mp match {
      case Nil => input
      case ::(h, t) =>
        h.foreach(m => tb.diffTheta(m, input.head))
        backPropThetaChannel(t, tb, input.tail)
    }
  }

  @tailrec
  final def backPropThetaFilter(tb: ThetaBucket, input: List[Double]): Unit = {
    input match {
      case Nil =>
      case _ =>
        val channeled = backPropThetaChannel(maps, tb, input)
        backPropThetaFilter(tb, channeled)
    }
  }

  @tailrec
  private def back(i: Int, d: Array[Double], grads: Array[Array[Double]], input: List[Double]): Unit = {
    if (i == grads.length) return
    val diffs = d.take(filterSize)
    grads(i)(0) += diffs.sum // Bias node
    backPropThetaFilter(ThetaBucket(grads(i), diffs, Array.fill(filterSize)(1)), input)
    back(i+1, d.drop(filterSize), grads, input)
  }

  override def getGradients(currentDiff: List[Double], prevActs: List[Double], grads: Array[Array[Double]]): Array[Array[Double]] = {
    back(0, currentDiff.toArray, grads, prevActs)
    grads
  }

}
