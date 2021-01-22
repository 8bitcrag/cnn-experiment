package com.eightbit85.monnn

import cats.data.Reader
import cats.instances.double._
import cats.syntax.semigroup._
import com.eightbit85.monnn.cate.ErrorState
import com.eightbit85.monnn.cate.Implicits._
import com.eightbit85.monnn.GradientChecking._
import com.eightbit85.monnn.helpers.MatrixHelpers._
import com.eightbit85.monnn.helpers.SeqHelpers._
import com.eightbit85.monnn.helpers.{SeqHelper, Util}

import scala.annotation.tailrec

object ProcessingActions {

  private def getCost(features: List[TrainingPoint], size: Int, thetas: List[ThetaLayer]): Double = {
    val interpreter = NeuralNetwork.compiler(thetas)
    val total = features.foldLeft(0.0)((a, f) => {
      val result = for {
        fwd <- NeuralNetwork.forwardProp(f.o)
        cst <- NeuralNetwork.cost(fwd.head, f.y)
      } yield a + cst
      result.foldMap(interpreter)
    })
    total / size
  }

  private def thetasToGradients(conf: Config, thetas: List[ThetaLayer]): List[List[List[Double]]] = {
    val interpreter = NeuralNetwork.compiler(thetas)
    val learnPoint: Reader[(List[Array[Array[Double]]], TrainingPoint), List[Array[Array[Double]]]] =
      Reader(in => {
        val process = for {
          fp <- NeuralNetwork.forwardProp(in._2.o)
          fd <- NeuralNetwork.firstDifferential(fp, in._2.y)
          gr <- NeuralNetwork.backProp(fd, in._1)
        } yield gr

        process.foldMap(interpreter)
      })

    conf.features
      .foldLeft(thetas.map(_.gradientPlaceHolders))((g, n) => learnPoint.run((g, n)))
      .map(gr => {
        gr.map(row => {
          row.map(t => conf.alpha * (t / conf.size)).toList
        }).toList
      })
  }

  private def iterateToLimit(conf: Config, thetas: List[ThetaLayer]): List[ThetaLayer] = {
    val gradientThreshold: List[List[List[Double]]] => Boolean = gradientsToSizeCheck(conf.threshold)

    @tailrec
    def limitedIterations(t: List[ThetaLayer], it: Int): List[ThetaLayer] = {
      val gradients = thetasToGradients(conf, t)
      if (!gradientThreshold(gradients)) t
      else {
        val updated = t.zip(gradients).map({
          case (Normal(th), g) => Normal(th |+| g)
          case (Convolutional(th, mp, dp, fs), g) => Convolutional(th |+| g, mp, dp, fs)
        })
        if (it == conf.limit) updated
        else {
          Util.impureLog(s"iteration $it complete")
          limitedIterations(updated, it+1)
        }
      }
    }

    limitedIterations(thetas, 1)
  }

  val startingPoint: Config => ErrorState[List[ThetaLayer], Double] = conf => ErrorState[List[ThetaLayer], Double] { thetas =>
    Right((thetas, getCost(conf.features, conf.size, thetas)))
  }

  val gradientCheck: Config => ErrorState[List[ThetaLayer], Double] = conf => ErrorState[List[ThetaLayer], Double] { thetas =>
    // forward prop morphisms from different starting points
    val chains = composeChain(thetas.reverse.toArray, 0, new Array[List[List[Double]] => List[List[Double]]](thetas.length))

    // Normal learning for one training point
    val interpreter = NeuralNetwork.compiler(thetas)
    val compute = for {
      fp <- NeuralNetwork.forwardProp(conf.features.head.o)
      fd <- NeuralNetwork.firstDifferential(fp, conf.features.head.y)
      gr <- NeuralNetwork.backProp(fd, thetas.map(_.gradientPlaceHolders))
    } yield (fp.reverse.toArray, gr.toArray)
    val preCalc = compute.foldMap(interpreter)

    // forward prop functions for specific nodes at different points in the network
    val fp = thetas.map({
      case Normal(_) => (t: Array[Double], i: List[Double]) => List(sigmoid(SeqHelper[Array].elemProdSum(t, 1 :: i)))
      case c@Convolutional(_, maps, _, sz) => (t: Array[Double], i: List[Double]) => c.fpf(t, i)
    }).toArray

    allButFinalLayerGradCheck(thetas.toArray, preCalc._2, preCalc._1, chains, fp, conf.features.head.y, conf.epsilon)
      .flatMap(_ => finalLayerGradCheck(thetas.last.thetas.map(_.toArray).toArray, preCalc._2.last, preCalc._1.reverse.tail.head, conf.features.head.y, conf.epsilon))
      .map(_ => (thetas, 0.0))
  }

  val learn: Config => ErrorState[List[ThetaLayer], Double] = conf => ErrorState[List[ThetaLayer], Double] { thetas =>
    val t = Util.time("it", ProcessingActions.iterateToLimit(conf, thetas))
    Right((t, ProcessingActions.getCost(conf.features, conf.size, t)))
  }

  val compare: Config => ErrorState[List[ThetaLayer], Double] = conf => ErrorState[List[ThetaLayer], Double] { thetas =>
    val interpreter = NeuralNetwork.compiler(thetas)
    val total = conf.features.foldLeft((0, 0.0))((a, f) => {
      val calc = for {
        fp <- NeuralNetwork.forwardProp(f.o)
        cst <- NeuralNetwork.cost(fp.head, f.y)
      } yield (fp.head, fp.head.indexOf(fp.head.max), a._2 + cst)
      val results = calc.foldMap(interpreter)

      val isCorrect = if (f.y.indexOf(1) == results._2) 1 else 0
      //      println(isCorrect)
      (a._1 + isCorrect, results._3)
    })
    val cst = total._2 / conf.size
    val error = (conf.size - total._1).toDouble / conf.size.toDouble
    Util.impureLog(total._1 + " of " + conf.size + " correct. " + error + " error.")
    Right((thetas, cst))
  }

}
