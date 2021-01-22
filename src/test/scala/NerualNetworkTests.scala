package com.eightbit85.monnn

import cats.data.Reader
import com.eightbit85.monnn.helpers.SeqHelper
import com.eightbit85.monnn.helpers.SeqHelpers.listSeqHelper
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Prop.forAll
import org.scalacheck.{Gen, Test}

class NeuralNetworkTests extends munit.DisciplineSuite {

  override def scalaCheckTestParameters: Test.Parameters = super.scalaCheckTestParameters
    .withMinSize(2)
    .withMaxSize(10)

  def matrix: Gen[Normal] = Gen.sized { size =>
    Gen.listOfN(size, Gen.listOfN(size, arbitrary[Double])).map(f => Normal(f))
  }

  def matrix2: Gen[(Normal, Normal)] = for {
    m1 <- matrix
    m2 <- matrix
  } yield (m1, m2)

  val matrixAndVec: Gen[(List[List[Double]], List[Double])] = Gen.sized { size =>
    val side = scala.math.sqrt(size).asInstanceOf[Int]
    for {
      m <- Gen.listOfN(side, Gen.listOfN(side, arbitrary[Double]))
      n <- Gen.listOfN(side, arbitrary[Double])
    } yield (m, n)
  }

  val vec: Gen[List[Double]] = Gen.sized { size =>
    Gen.listOfN(size, arbitrary[Double])
  }

  val vecAndY: Gen[(List[Double], Double)] = for {
    v <- Gen.nonEmptyListOf(arbitrary[Double])
    y <- Gen.oneOf(0.0, 1.0)
  } yield (v, y)

  property("NeuralNetwork.partMemoizedActivations combines a series of activation functions in to one") {
    forAll(matrix2) { case(m1, m2) =>
      val input: List[Double] = List.fill(m1.thetas.head.length)(1.0)
      val l1 = SeqHelper[List].toActivations(m1.thetas)(input)
      val l2 = SeqHelper[List].toActivations(m2.thetas)(l1)

      val mem = NeuralNetwork.partAppliedActivations(List(m1, m2))
      val res = mem(List(input))

      assertEquals(res, List(l2, l1, input))
    }
  }

  test("Forward Propagation") {
    val thetas: List[Normal] = List(
      Normal(List(
        List(1.0, -1.0, -1.0, 0.0),
        List(-1.0, 0.0, 0.0, 2.0)
      )),
      Normal(List(
        List(-2.5, 2.0, 2.0),
        List(1.3, -2.0, -2.0)
      )),
      Normal(List(
        List(-0.73, 1.0, 1.0)
      ))
    )

    val fhelp = List(0.0, 1.0)

    val features = for {
      i <- fhelp
      j <- fhelp
      k <- fhelp
    } yield TrainingPoint(List(List(i, j, k)), if ((i+j == 2 && k == 0) || (i+j == 0 && k == 1)) List(1) else List(0))

    val comp = NeuralNetwork.compiler(thetas)
    val rd: Reader[List[List[Double]], List[Double]] = Reader(in => {
      val out = for (fp <- NeuralNetwork.forwardProp(in)) yield fp.head
      out.foldMap(comp)
    })

    val results = features.map(nl => {
      val forwardProp = rd.run(nl.o)

      nl.y.head match {
        case 1 => forwardProp.head >= 0.5
        case 0 => forwardProp.head < 0.5
      }
    })

    assert(results.forall(r => r))
  }

}
