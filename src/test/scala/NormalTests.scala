package com.eightbit85.monnn

import com.eightbit85.monnn.helpers.MatrixHelpers.sigmoid
import com.eightbit85.monnn.helpers.SeqHelper
import com.eightbit85.monnn.helpers.SeqHelpers.listSeqHelper
import org.scalacheck.Prop.forAll
import org.scalacheck.{Gen, Test}
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._

class NormalTests extends munit.DisciplineSuite {

  override def scalaCheckTestParameters: Test.Parameters =
    super.scalaCheckTestParameters
      .withMinSize(2)

  def smallValue: Gen[Double] = Gen.choose(0.0, 1.0)

  val doublesGen: Gen[(Double, Double, Double)] = for {
    k <- smallValue
    l <- smallValue
    m <- smallValue
  } yield (k, l, m)

  val testGen: Gen[(Double, Double, Double, Double, Double, Double, Double, Double)] = for {
    k <- smallValue
    l <- smallValue
    m <- smallValue
    n <- smallValue
    o <- smallValue
    p <- smallValue
    q <- Gen.choose(1.0, 15.0)
    r <- Gen.choose(1.0, 15.0)
  } yield (k, l, m, n, o, p, q, r)

  def bpGen: Gen[(List[Double], List[List[Double]], List[Double])] = for {
    in <- Gen.listOfN(10, smallValue)
    th <- Gen.listOfN(5, Gen.listOfN(11, smallValue.map(_ * 2 - 1)))
    df <- Gen.listOfN(5, smallValue)
  } yield (in, th, df)

  def sizedBpGen: Gen[(List[Double], List[List[Double]], List[Double])] = Gen.sized { size =>
    for {
      in <- Gen.listOfN(size, smallValue)
      th <- Gen.listOfN(size-1, Gen.listOfN(size+1, smallValue.map(_ * 2 - 1))) // Descending network, +1 for bias
      df <- Gen.listOfN(size-1, smallValue)
    } yield (in, th, df)
  }

  def matrixAndVec[T](g: Gen[T]): Gen[(List[List[T]], List[T])] = Gen.sized { size =>
    for {
      m <- Gen.listOfN(size, Gen.listOfN(size, g))
      n <- Gen.listOfN(size, g)
    } yield (m, n)
  }

  property("Normal.liftActivations adds the new layer of activations to the stack") {
    forAll(matrixAndVec(Gen.choose(-3.0, 3.0))) { case (m, v) =>
      val tl = Normal(m)
      val act = SeqHelper[List].toActivations(m)(v)
      val res = tl.liftActivations(List(v))
      assertEquals(res.head, act)
    }
  }

  property("Normal.getNextDiffWithRespectToNodes combines a layer diff with previous thetas to get previous layer partial diff (not including act func diff)") {
    forAll(bpGen) { case (in, th, df) =>
      val tl = Normal(th)
      val res = tl.getNextDiffWithRespectToNodes(df)

      val exp1 = df.zip(th).map(t => t._1 * t._2(1)).sum
      val exp2 = df.zip(th).map(t => t._1 * t._2(2)).sum
      val exp3 = df.zip(th).map(t => t._1 * t._2(3)).sum
      assertEquals(res.head, exp1)
      assertEquals(res(1), exp2)
      assertEquals(res(2), exp3)
    }
  }

  property("Normal.getGradients combines a layer diff with previous activation layer to produce gradients") {
    forAll(bpGen) { case (in, th, df) =>
      val tl = Normal(th)
      val grads = Array.fill(5)(Array.fill(11)(0.0))
      val res = tl.getGradients(df, in, grads)

      assertEquals(grads(0)(0), df.head)
      assertEquals(grads(3)(8), df(3) * in(7))
      assertEquals(grads(4)(0), df(4))
    }
  }

  property("Normal.getGradients produces correct size image") {
    forAll(sizedBpGen) { case (in, th, df) =>
      val grads = Array.fill(th.length)(Array.fill(th.head.length)(0.0))
      val tl = Normal(th)
      val bp = tl.getGradients(df, in, grads)
      assertEquals(bp.length, df.length) // No. of rows matches No. of outputs
      assertEquals(bp.head.length, in.length + 1) // No. cols matches No. of thetas in a row
    }
  }

  property("Normal.getGradients calculates correct gradients") {
    forAll(sizedBpGen) { case (in, th, df) =>
      val grads = Array.fill(th.length)(Array.fill(th.head.length)(0.0))
      val tl = Normal(th)
      val bp = tl.getGradients(df, in, grads)
      val shouldBe = df.head * in.head
      assertEquals(bp.head(1), shouldBe)
    }
  }

  property("Normal.getNextDiffWithRespectToNodes activation differentials are correct") {
    forAll(doublesGen) { case (k, l, m) =>
      val tl = Normal(List(List(5.0, -1.0, 1.5)))
      val grads = Array.fill(1)(Array.fill(3)(0.0))
      val bp = tl.getNextDiffWithRespectToNodes(List(k))
      val shouldBeA = -1.0 * k
      val shouldBeB = 1.5 * k
      assert(bp.head === shouldBeA +- 0.000001)
      assert(bp(1) === shouldBeB +- 0.000001)
    }
  }

  property("Normal.gradientChecking backProp diffs match first principle diffs") {
    forAll(testGen) { case (t1, t2, t3, t4, t5, t6, i1, i2) =>
      val epsilon = 0.0001
      val y = List(1.0, 0.0)
      val tl = Normal(List(List(t1, t2, t3), List(t4, t5, t6)))

      // applying weights to input
      val fpMinus = List(sigmoid(t1 + (t2 - epsilon) * i1 + t3 * i2), sigmoid(t4 + t5 * i1 + t6 * i2))
      val fpOrig = List(sigmoid(t1 + t2 * i1 + t3 * i2), sigmoid(t4 + t5 * i1 + t6 * i2))
      val fpPlus = List(sigmoid(t1 + (t2 + epsilon) * i1 + t3 * i2), sigmoid(t4 + t5 * i1 + t6 * i2))

      // approx gradient
      val manual = (NeuralNetwork.multiclassCost(y)(fpPlus) - NeuralNetwork.multiclassCost(y)(fpMinus)) / (2 * epsilon)

      val initDiff = fpOrig.zip(y).map(t => t._1 - t._2)
      val grads1 = Array.fill(tl.thetas.length)(Array.fill(tl.thetas.head.length)(0.0))
      val bp = tl.getGradients(initDiff, List(i1, i2), grads1) // NormalOps.backProp(grads1).run(layer).value

      assert(
        bp.head(1) === manual +- epsilon,
        "The calculated theta differential was not close enough to the manual differential"
      )
    }
  }

}
