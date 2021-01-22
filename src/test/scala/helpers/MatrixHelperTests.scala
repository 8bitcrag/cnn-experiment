package com.eightbit85.monnn.helpers

import cats.instances.double._
import com.eightbit85.monnn.helpers.MatrixHelpers._
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Prop._
import org.scalacheck.{Gen, Test}

class MatrixHelperTests extends munit.DisciplineSuite {

  override def scalaCheckTestParameters: Test.Parameters = super.scalaCheckTestParameters
    .withMinSize(2)
    .withMaxSize(10)

  test("MatrixHelpers.gradientsToSizeCheck is true when gradients are too big to stop") {
    val grads = List(
      List(
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.0001, 0.0001)
      ),
      List(
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.01, 0.0001),
        List(0.0001, 0.0001, 0.0001)
      )
    )

    assert(gradientsToSizeCheck(0.001)(grads))
  }

  test("MatrixHelpers.gradientsToSizeCheck is false when gradients are too small to carry on") {
    val grads = List(
      List(
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.0001, 0.0001)
      ),
      List(
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.0001, 0.0001),
        List(0.0001, 0.0001, 0.0001)
      )
    )

    assert(!gradientsToSizeCheck(0.0001)(grads))
  }

  test("MatrixHelpers.sigmoid maps numbers to the sigmoid curve") {
    val a = sigmoid(0)
    val b = sigmoid(-1)
    val c = sigmoid(1)

    assertEquals(f"$a%1.4f", "0.5000")
    assertEquals(f"$b%1.4f", "0.2689")
    assertEquals(f"$c%1.4f", "0.7311")
  }

  def matrix[T](g: Gen[T]): Gen[List[List[T]]] = Gen.sized { size =>
    Gen.listOfN(size, Gen.listOfN(size, g))
  }

  property("MatrixHelpers.transpose flips matrices") {
    forAll(matrix(arbitrary[Double])) { mtx =>
      val trans = transpose(mtx)
      val expectedRow = mtx.map(r => r.head)
      assertEquals(trans.head, expectedRow)
    }
  }

  test("MatrixHelpers.buildMegaSequence generates a sequence of mappings from an input position to a positions on a filter") {
    val ms = buildMegaSequence(29, 0, 0, -1, 13, Nil)
    assertEquals(ms.length, 841)
    assertEquals(ms.head.head, 0)
    assertEquals(ms(14), List(5, 6, 7))
    assertEquals(ms(72), List(5, 6, 7, 18, 19, 20))
    assertEquals(ms(840).length, 1)
    assertEquals(ms(840).head, 168)
  }

}
