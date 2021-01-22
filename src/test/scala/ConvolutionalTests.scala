package com.eightbit85.monnn

import com.eightbit85.monnn.helpers.MatrixHelpers.{buildMegaSequence, sigmoid}
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll
import org.scalactic.Tolerance._
import org.scalactic.TripleEquals._

class ConvolutionalTests extends munit.DisciplineSuite {

  val m1: List[List[Int]] = buildMegaSequence(29, 0, 0, -1, 13, Nil)
  val m2: List[List[Int]] = buildMegaSequence(13, 0, 0, -1, 5, Nil)

  def valueGen: Gen[Double] = Gen.choose(0.0, 1.0)

  def bucketGen: Gen[ThetaBucket] = for {
    th <- Gen.containerOfN[Array, Double](5*5*5+1, valueGen)
    re <- Gen.containerOfN[Array, Double](5*5, th(0))
    co <- Gen.containerOfN[Array, Int](5*5, 1)
  } yield ThetaBucket(th, re, co)

  def bucketBackGen: Gen[ThetaBucket] = for {
    th <- Gen.containerOfN[Array, Double](5*5*5+1, valueGen)
    re <- Gen.containerOfN[Array, Double](5*5, valueGen)
    co <- Gen.containerOfN[Array, Int](5*5, 1)
  } yield ThetaBucket(th, re, co)

  def bucketThetaGen: Gen[ThetaBucket] = for {
    th <- Gen.containerOfN[Array, Double](5*5*5+1, 0.0)
    re <- Gen.containerOfN[Array, Double](5*5, valueGen)
    co <- Gen.containerOfN[Array, Int](5*5, 1)
  } yield ThetaBucket(th, re, co)

  def inputGen: Gen[List[Double]] = Gen.listOfN(13*13*5, valueGen)

  def fpcGen: Gen[(ThetaBucket, List[Double])] = for {
    t <- bucketGen
    i <- inputGen
  } yield (t, i)

  def bptGen: Gen[(ThetaBucket, List[Double])] = for {
    t <- bucketThetaGen
    i <- inputGen
  } yield (t, i)

  def allGen: Gen[(List[Double], List[List[Double]], List[Double])] = for {
    in <- inputGen
    th <- Gen.listOfN(3, Gen.listOfN(126, valueGen))
    df <- Gen.listOfN(5*5*3, valueGen)
  } yield (in, th, df)

  property("Convolutional.forwardPropChannel combines a single channel of input with its corresponding thetas") {
    forAll(fpcGen) { case (th, in) =>
      val tl = Convolutional(List(th.thetas.toList), m2, 1, 25)
      val getWindow: Int => List[Double] => List[Double] = s => i =>
        i.slice(s, s+5) ::: i.slice(s+13, s+18) ::: i.slice(s+26, s+31) ::: i.slice(s+39, s+44) ::: i.slice(s+52, s+57)
      val res = tl.forwardPropChannel(m2, th, in)

      val win1 = getWindow(0)(in)
      val win2 = getWindow(2)(in)
      val win3 = getWindow(4)(in)
      val win4 = getWindow(6)(in)
      val win5 = getWindow(8)(in)

      val exp1 = win1.zip(th.thetas.drop(1)).map(t => t._1 * t._2).sum + th.thetas(0)
      val exp2 = win2.zip(th.thetas.drop(1)).map(t => t._1 * t._2).sum + th.thetas(0)
      val exp3 = win3.zip(th.thetas.drop(1)).map(t => t._1 * t._2).sum + th.thetas(0)
      val exp4 = win4.zip(th.thetas.drop(1)).map(t => t._1 * t._2).sum + th.thetas(0)
      val exp5 = win5.zip(th.thetas.drop(1)).map(t => t._1 * t._2).sum + th.thetas(0)

      assertEquals(res.length, 13*13*4)
      assertEquals(res.head, in(169))
      assert(th.result(0) === exp1 +- 0.0000001)
      assert(th.result(1) === exp2 +- 0.0000001)
      assert(th.result(2) === exp3 +- 0.0000001)
      assert(th.result(3) === exp4 +- 0.0000001)
      assert(th.result(4) === exp5 +- 0.0000001)
    }
  }

  property("Convolutional.forwardPropFilter combines an entire activation layer with a filter") {
    forAll(fpcGen) { case (th, in) =>
      val tl = Convolutional(List(th.thetas.toList), m2, 1, 25)
      val getWindow: Int => List[Double] => List[Double] = s => i =>
        i.slice(s, s+5) ::: i.slice(s+13, s+18) ::: i.slice(s+26, s+31) ::: i.slice(s+39, s+44) ::: i.slice(s+52, s+57)

      val win1 = getWindow(0)(in)
      val win2 = getWindow(169)(in)
      val win3 = getWindow(338)(in)
      val win4 = getWindow(507)(in)
      val win5 = getWindow(676)(in)

      val exp1 = (win1 ::: win2 ::: win3 ::: win4 ::: win5).zip(th.thetas.drop(1))
        .map(t => t._1 * t._2).sum + th.thetas(0)

      val res = tl.forwardPropFilter(th, in)

      assertEquals(res.length, 25)
      assert(res.head === sigmoid(exp1) +- 0.0000001)
    }
  }

  property("Convolutional.getNextDiffWithRespectToNodes combines a layer diff with previous thetas to get previous layer partial diff (not including act func diff)") {
    forAll(allGen) { case (_, th, df) =>
      val tl = Convolutional(th, m2, 5, 25)
      val res = tl.getNextDiffWithRespectToNodes(df)
      val a: Int => Int => Double = f => s => {
        IndexedSeq(125, 123, 121, 115, 113, 111, 105, 103, 101)
          .zip(IndexedSeq(12, 13, 14, 17, 18, 19, 22, 23, 24).map(_ + s))
          .map(t => th(f)(t._1) * df(t._2))
          .sum
      }

      val exp1 = th.head(1) * df.head + th(1)(1) * df(25) + th(2)(1) * df(50)
      val exp2 = th(0)(3) * df(0) + th(0)(1) * df(1) +
        th(1)(3) * df(25) + th(1)(1) * df(26) +
        th(2)(3) * df(50) + th(2)(1) * df(51)
      val exp3 = th(0)(125) * df(24) + th(1)(125) * df(49) + th(2)(125) * df(74)
      val exp4 = a(0)(0) + a(1)(25) + a(2)(50)
      val got1 = res(844)
      val got2 = res(788)
      assert(res.head === exp1 +- 0.0000001)
      assert(res(2) === exp2 +- 0.0000001)
      assert(got1 === exp3 +- 0.0000001)
      assert(got2 === exp4 +- 0.0000001)
      assertEquals(res.length, 845)
    }
  }

  property("Convolutional.backPropThetaChannel combines a layer diff with a channel of a previous activation layer to produce gradients for part of a filter") {
    forAll(bptGen) { case (bu, in) =>
      val tl = Convolutional(Nil, m2, 5, 25)
      bu.thetas(0) += bu.result.sum

      val mini = (0 until 5).map(_ * 2)
      val exp1 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      }).zip(0 to 24)
        .map(t => in(t._1) * bu.result(t._2))
        .sum

      tl.backPropThetaChannel(m2, bu, in)

      assertEquals(bu.thetas(1), exp1)
    }
  }

  property("Convolutional.backPropThetaFilter combines a layer diff with a previous activation layer to produce gradients for a filter") {
    forAll(bptGen) { case (bu, in) =>
      val tl = Convolutional(Nil, m2, 5, 25)
      bu.thetas(0) += bu.result.sum

      val mini = (0 until 5).map(_ * 2)
      val exp1 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      }).zip(0 to 24)
        .map(t => in(t._1) * bu.result(t._2))
        .sum

      val exp2 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      })
        .map(_ + 169)
        .zip(0 to 24)
        .map(t => in(t._1) * bu.result(t._2))
        .sum

      val exp3 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      })
        .map(_ + 676 + 56)
        .zip(0 to 24)
        .map(t => in(t._1) * bu.result(t._2))
        .sum

      tl.backPropThetaFilter(bu, in)

      assertEquals(bu.thetas(1), exp1)
      assertEquals(bu.thetas(26), exp2)
      assertEquals(bu.thetas(125), exp3)
    }
  }

  property("Convolutional.backPropIntegrationCheck back propagation should work as a whole") {
    forAll(allGen) { case (in, th, df) =>
      val grads = Array.fill(3)(Array.fill(126)(0.0))

      val tl = Convolutional(th, m2, 5, 25)
      val resG = tl.getGradients(df, in, grads)
      val resN = tl.getNextDiffWithRespectToNodes(df)

      val mini = (0 until 5).map(_ * 2)
      val exp1 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      }).zip(0 to 24)
        .map(t => in(t._1) * df(t._2))
        .sum

      val exp2 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      })
        .map(_ + 169)
        .zip(25 to 49)
        .map(t => in(t._1) * df(t._2))
        .sum

      // last channel starts at 676 (13*13 input size, 5 channels)
      // last theta position - 5th row starts at 4*13=52, move in 4 spots to 56
      // last theta is 125 - 5*5 filter, 5 channels deep, 1 bias node = 126 (125 index)
      // last filter so last set of diffs. 3 filters of 25, so last starts at 50
      val exp3 = (0 until 5).flatMap(i => {
        mini.map(_ + i*26)
      })
        .map(_ + 676 + 56)
        .zip(50 to 74)
        .map(t => in(t._1) * df(t._2))
        .sum

      assertEquals(grads(0)(1), exp1)
      assertEquals(grads(1)(26), exp2)
      assertEquals(grads(2)(125), exp3)

      assertEquals(resG(0)(1), exp1)
      assertEquals(resG(1)(26), exp2)
      assertEquals(resG(2)(125), exp3)

      // check node grads

      val a: Int => Int => Double = f => s => {
        IndexedSeq(125, 123, 121, 115, 113, 111, 105, 103, 101)
          .zip(IndexedSeq(12, 13, 14, 17, 18, 19, 22, 23, 24).map(_ + s))
          .map(t => th(f)(t._1) * df(t._2))
          .sum
      }

      val exp4 = (a(0)(0) + a(1)(25) + a(2)(50))
      val got = resN(788)
      assert(got === exp4 +- 0.000001)
    }
  }

  def initTheta(in: Int, out: Int): Gen[Double] = Gen.choose(0.0, 1.0)
    .map(n => {
      val ep = Math.sqrt(6) / math.sqrt(in + out)
      (2 * ep) * n - ep
    })

  def thetaGen: Gen[List[ThetaLayer]] = for {
    t1 <- Gen.listOfN(5, Gen.listOfN(26, initTheta(841, 845)))
    t2 <- Gen.listOfN(50, Gen.listOfN(126, initTheta(845, 250)))
    t3 <- Gen.listOfN(100, Gen.listOfN(1251, initTheta(1250, 100)))
    t4 <- Gen.listOfN(10, Gen.listOfN(101, initTheta(100, 10)))
  } yield List(
    Convolutional(t1, m1, 1, 169),
    Convolutional(t2, m2, 5, 25),
    Normal(t3),
    Normal(t4)
  )

  def checkGen: Gen[(List[List[Double]], List[List[Double]], List[ThetaLayer], Int, Int)] = for {
    th <- thetaGen
    in <- Gen.listOfN(29*29, valueGen)
    inb <- Gen.listOfN(29*29, valueGen)
    y <- Gen.choose(0, 9)
    yb <- Gen.choose(0, 9)
  } yield (List(in), List(inb), th, y, yb)

  property("Convolutional.gradientCheck backprop gradients should match *sum* of first principle gradients. To be averaged later.") {
    forAll(checkGen) { case (inA, inB, th, yA, yB) =>
      val interpreter = NeuralNetwork.compiler(th)
      val epsilon = 0.0001
      val grads = List(
        Array.fill(5)(Array.fill(26)(0.0)),
        Array.fill(50)(Array.fill(126)(0.0)),
        Array.fill(100)(Array.fill(1251)(0.0)),
        Array.fill(10)(Array.fill(101)(0.0))
      )

      def doTest(in: List[List[Double]], y: Int): Double = {
        val actual = (0 to 9).map(i => if (i == y) 1.0 else 0.0).toList

        val calc = for {
          fp <- NeuralNetwork.forwardProp(in)
          fd <- NeuralNetwork.firstDifferential(fp, actual)
          gr <- NeuralNetwork.backProp(fd, grads)
        } yield ()
        calc.foldMap(interpreter)

        val first = th.head.asInstanceOf[Convolutional]
        val originalTheta = first.thetas.head.head

        val plus = first.thetas.head.updated(0, originalTheta + epsilon)
        val thPlus = first.copy(thetas = first.thetas.updated(0, plus))
        val hypPlus = NeuralNetwork.partAppliedActivations(thPlus :: th.tail)(in)
        val costPlus = NeuralNetwork.multiclassCost(actual)(hypPlus.head)

        val minus = first.thetas.head.updated(0, originalTheta - epsilon)
        val thMinus = first.copy(thetas = first.thetas.updated(0, minus))
        val hypMinus = NeuralNetwork.partAppliedActivations(thMinus :: th.tail)(in)
        val costMinus = NeuralNetwork.multiclassCost(actual)(hypMinus.head)

        (costPlus - costMinus) / (2 * epsilon)
      }

      val man1 = doTest(inA, yA)
      val man2 = doTest(inB, yB)

      assert(grads.head(0)(0) === (man1 + man2) +- epsilon)
    }
  }

}
