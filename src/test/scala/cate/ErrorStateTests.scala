package com.eightbit85.monnn.cate

import cats.Eq
import cats.implicits._
import cats.laws.discipline.arbitrary.catsLawsArbitraryForMiniInt
import cats.laws.discipline.{MiniInt, MonadTests}
import org.scalacheck.Arbitrary
import org.scalacheck.Arbitrary.arbitrary

class ErrorStateTests extends munit.DisciplineSuite {

  implicit def errorStateEq[A: Eq]: Eq[ErrorState[MiniInt, A]] = Eq.instance[ErrorState[MiniInt, A]] { (es1, es2) =>
    arbitrary[MiniInt].sample.exists(i => {
      es1.run(i) == es2.run(i)
    })
  }
  implicit def arbErrorState[A: Arbitrary]: Arbitrary[ErrorState[MiniInt, A]] =
    Arbitrary(for {
      a <- arbitrary[A]
    } yield ErrorState[MiniInt, A](s => Right((s, a)) ))

  def testA: ErrorState[String, Double] = ErrorState[String, Double] { st =>
    if (st == "err") Left(new Exception("testA"))
    else Right((st + " test A", 2.0))
  }

  def testB: ErrorState[String, Double] = ErrorState[String, Double] { st =>
    if (st == "err") Left(new Exception("testB"))
    else Right((st + " test B", 2.0))
  }

  def testC: ErrorState[String, Double] = ErrorState[String, Double] { st =>
    if (st == "err") Left(new Exception("testC"))
    else Right((st + " test C", 2.0))
  }

  checkAll("ErrorState.MonadLaws", MonadTests[({type L[a] = ErrorState[MiniInt, a]})#L].monad[Int, Int, Int])

  test("ErrorState.composes a sequence of ErrorStates should compose") {
    val x = for {
      a <- testA
      b <- testB
      c <- testC
    } yield (a, b, c)

    x.run("start") match {
      case Left(_) => fail("ErrorState did not compose")
      case Right((s, c)) => assertEquals(s, "start test A test B test C")
    }

  }

  test("ErrorState.shortCircuit a sequence of ErrorStates should short circuit if an error is found") {
    val x = for {
      a <- testA
      b <- testB
      c <- testC
    } yield (a, b, c)

    x.run("err") match {
      case Left(e) => assertEquals(e.getMessage, "testA")
      case Right((s, c)) => fail("ErrorState did not short circuit")
    }
  }

}
