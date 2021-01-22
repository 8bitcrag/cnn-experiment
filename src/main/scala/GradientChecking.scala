package com.eightbit85.monnn

import com.eightbit85.monnn.helpers.SeqHelper
import com.eightbit85.monnn.helpers.SeqHelpers.jarraySeqHelper
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._

import scala.annotation.tailrec

object GradientChecking {

  @tailrec
  final def composeChain(thetas: Array[ThetaLayer], idx: Int, result: Array[List[List[Double]] => List[List[Double]]]): Array[List[List[Double]] => List[List[Double]]] = {
    if (idx == thetas.length) return result.reverse
    val current = thetas(idx).liftActivations _
    val comp = if (idx > 0) result(idx - 1) compose current else current
    result(idx) = comp
    composeChain(thetas, idx+1, result)
  }

  def allButFinalLayerGradCheck(
                                 thetaLayers: Array[ThetaLayer],
                                 calculated: Array[Array[Array[Double]]],
                                 precalc: Array[List[Double]],
                                 chains: Array[List[List[Double]] => List[List[Double]]],
                                 fp: Array[(Array[Double], List[Double]) => List[Double]],
                                 actual: List[Double],
                                 epsi: Double
                               ): Either[Exception, Int] = {
    (0 until thetaLayers.length - 1).firstError { t =>
      val offset = thetaLayers(t) match {
        case Convolutional(_, _, _, fs) => fs
        case _ => 1
      }
      val thetas = thetaLayers(t).thetas.map(_.toArray).toArray
      val mtx = thetas.indices.firstError { r =>
        val row = thetas(r).indices.firstError { c =>
          val original: Double = thetas(r)(c)
          thetas(r).update(c, original + epsi)
          val updated = fp(t)(thetas(r), precalc(t))
          val l1Plus = precalc(t+1).patch(r*offset, updated, offset)
          val plus = chains(t + 1)(List(l1Plus)).head
          thetas(r).update(c, original - epsi)
          val l1Minus = precalc(t+1).patch(r*offset, fp(t)(thetas(r), precalc(t)), offset)
          val minus = chains(t + 1)(List(l1Minus)).head
          thetas(r).update(c, original)
          val gr = (NeuralNetwork.multiclassCost(actual)(plus) - NeuralNetwork.multiclassCost(actual)(minus)) / (2 * epsi)
          if (calculated(t)(r)(c) === gr +- 0.0001) Right(1)
          else Left(s"$c")
        }

        row match {
          case Left(i) => Left(s"$r, $i")
          case Right(i) => Right(i)
        }
      }

      mtx match {
        case Left(i) => Left(new Exception(s"Gradient check failed: $t, $i"))
        case Right(i) => Right(i)
      }
    }
  }

  def finalLayerGradCheck(
                           thetas: Array[Array[Double]],
                           calculated: Array[Array[Double]],
                           precalc: List[Double],
                           actual: List[Double],
                           epsi: Double
                         ): Either[Exception, Int] = {

    thetas.indices.firstError { r =>
      val row = thetas(r).indices.firstError { c =>
        val original: Double = thetas(r)(c)
        thetas(r).update(c, original + epsi)
        val plus = SeqHelper[Array].toActivations(thetas)(precalc)
        thetas(r).update(c, original - epsi)
        val minus = SeqHelper[Array].toActivations(thetas)(precalc)
        thetas(r).update(c, original)
        val gr = (NeuralNetwork.multiclassCost(actual)(plus) - NeuralNetwork.multiclassCost(actual)(minus)) / (2 * epsi)
        if (calculated(r)(c) === gr +- 0.0001) Right(1)
        else Left(s"$c")
      }

      row match {
        case Left(i) => Left(new Exception(s"Gradient check failed: $r, $i"))
        case Right(i) => Right(i)
      }
    }

  }

}
