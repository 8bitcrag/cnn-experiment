package com.eightbit85.monnn

import cats.free.Free.liftF
import cats.{Id, ~>}
import com.eightbit85.monnn.helpers.MatrixHelpers.sigmoidDiff
import com.eightbit85.monnn.helpers.SeqHelper
import com.eightbit85.monnn.helpers.SeqHelperSyntax.SeqHelperOps
import com.eightbit85.monnn.helpers.SeqHelpers.listSeqHelper

import scala.annotation.tailrec

sealed trait NeuralNetwork[A]
case class ForwardProp(input: List[List[Double]]) extends NeuralNetwork[List[List[Double]]]
case class FirstDifferential(activations: List[List[Double]], actual: List[Double]) extends NeuralNetwork[List[List[Double]]]
case class BackProp(toBackProp: List[List[Double]], gradients: List[Array[Array[Double]]]) extends NeuralNetwork[List[Array[Array[Double]]]]
case class Cost(hypothesis: List[Double], actual: List[Double]) extends NeuralNetwork[Double]

object NeuralNetwork {

  def forwardProp(input: List[List[Double]]): NeuralNetworkF[List[List[Double]]] =
    liftF[NeuralNetwork, List[List[Double]]](ForwardProp(input))
  def firstDifferential(activations: List[List[Double]], actual: List[Double]): NeuralNetworkF[List[List[Double]]] =
    liftF[NeuralNetwork, List[List[Double]]](FirstDifferential(activations, actual))
  def backProp(toBackProp: List[List[Double]], gradients: List[Array[Array[Double]]]): NeuralNetworkF[List[Array[Array[Double]]]] =
    liftF[NeuralNetwork, List[Array[Array[Double]]]](BackProp(toBackProp, gradients))
  def cost(hypothesis: List[Double], actual: List[Double]): NeuralNetworkF[Double] =
    liftF[NeuralNetwork, Double](Cost(hypothesis, actual))

  val partAppliedActivations: List[ThetaLayer] => List[List[Double]] => List[List[Double]] =
    tl => ac => tl.map(t => t.liftActivations _).reduceLeft((f, g) => g compose f)(ac)

  val singleCost: Double => Double => Double = y => h =>
    -(y*math.log(h) + (1-y)*math.log(1 - h))

  val multiclassCost: List[Double] => List[Double] => Double = y => h => {
    y.zip(h).map(t => singleCost(t._1)(t._2)).sum
  }

  @tailrec
  private def backPropagation(thetas: List[ThetaLayer], nodes: List[List[Double]], grads: List[Array[Array[Double]]], result: List[Array[Array[Double]]] = Nil): List[Array[Array[Double]]] = {
    nodes.tail match {
      case Nil =>
        throw new IndexOutOfBoundsException("Next activation layer does not exist")
      case ::(h, Nil) =>
        thetas.head.getGradients(nodes.head, h, grads.head) :: result
      case ::(h, _) =>
        val da = thetas.head.getNextDiffWithRespectToNodes(nodes.head)
        val sig = h.map(sigmoidDiff)
        val dz = da.elemProd(sig)
        val gr = thetas.head.getGradients(nodes.head, h, grads.head)

        backPropagation(thetas.tail, dz :: nodes.tail.tail, grads.tail, gr :: result)
    }
  }

  def compiler(thetas: List[ThetaLayer]): NeuralNetwork ~> Id =
    new (NeuralNetwork ~> Id) {
      val toActivations: List[List[Double]] => List[List[Double]] = partAppliedActivations(thetas)
      val revThetas: List[ThetaLayer] = thetas.reverse
      override def apply[A](fa: NeuralNetwork[A]): Id[A] = {
        fa match {
          case ForwardProp(i) => toActivations(i)
          case FirstDifferential(a, y) => SeqHelper[List].elemMinus(a.head, y) :: a.tail
          case BackProp(fd, gr) => backPropagation(revThetas, fd, gr.reverse)
          case Cost(hyp, act) => multiclassCost(act)(hyp)
        }
      }
    }

}
