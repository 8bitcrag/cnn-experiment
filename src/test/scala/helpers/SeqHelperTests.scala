package com.eightbit85.monnn.helpers

import com.eightbit85.monnn.helpers.SeqHelpers.listSeqHelper
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class SeqHelperTests extends munit.DisciplineSuite {

  def matrixAndVec[T](g: Gen[T]): Gen[(List[List[T]], List[T])] = Gen.sized { size =>
    for {
      m <- Gen.listOfN(size, Gen.listOfN(size, g))
      n <- Gen.listOfN(size, g)
    } yield (m, n)
  }

  property("SeqHelper.mtxVecToProduct performs matrix/vector multiplication") {
    forAll { (k: Double, l: Double, m: Double) =>
      val mtx = List(
        List(k, l, m),
        List(4.0, 5.0, 6.0),
        List(7.0, 8.0, 9.0)
      )

      val vec = List(1.0, 2.0, 3.0)
      val result = SeqHelper[List].mtxVecToProduct(mtx)(vec)
      val shouldBe = 1.0 * k + 2.0 * l + 3.0 * m
      assertEquals(result.head, shouldBe)
      assertEquals(result(1), 32.0)
      assertEquals(result(2), 50.0)
    }
  }

  property("SeqHelper.mtxVecToProduct always produces correct size result") {
    forAll(matrixAndVec(arbitrary[Double])) { case(m, v) =>
      val prod = SeqHelper[List].mtxVecToProduct(m)(v)
      assertEquals(prod.length, v.length)
    }
  }

  test("SeqHelper.toActivations adds a bias node, gets the product and maps each element to the sigmoid curve") {
    val mtx = List(
      List(1.0, 2.0, -2.0),
      List(-8.0, -5.0, 6.0),
      List(4.0, -15.0, 9.0)
    )

    val vec = List(2.0, 3.0)
    val result = SeqHelper[List].toActivations(mtx)(vec)
    assertEquals(result.head.formatted("%1.4f"), "0.2689")
    assertEquals(result(1).formatted("%1.4f"), "0.5000")
    assertEquals(result(2).formatted("%1.4f"), "0.7311")
  }

}
