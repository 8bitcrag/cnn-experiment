package com.eightbit85.monnn
package helpers

import scala.annotation.tailrec

object SeqHelpers {

  implicit def listSeqHelper: SeqHelper[List] = new SeqHelper[List] {

    @tailrec
    private def list2(m: List[Double], v: List[Double], res: Double): Double = {
      m match {
        case Nil => res
        case ::(h, t) => list2(t, v.tail, res + h * v.head)
      }
    }

    @tailrec
    private def combi(x: List[Double], y: List[Double], z: List[Double]): List[Double] = {
      x match {
        case Nil => z.reverse
        case ::(h, t) => combi(t, y.tail, (h * y.head) :: z)
      }
    }

    @tailrec
    private def elminus(x: List[Double], y: List[Double], z: List[Double]): List[Double] = {
      x match {
        case Nil => z.reverse
        case ::(h, t) => elminus(t, y.tail, (h - y.head) :: z)
      }
    }

    override def elemProdSum(s1: List[Double], s2: List[Double]): Double =
      list2(s1, s2, 0.0)

    override def elemProd(s1: List[Double], s2: List[Double]): List[Double] =
      combi(s1, s2, Nil)

    override def elemMinus(s1: List[Double], s2: List[Double]): List[Double] =
      elminus(s1, s2, Nil)

  }

  implicit def jarraySeqHelper: SeqHelper[Array] = new SeqHelper[Array] {

    override def elemProdSum(s1: Array[Double], s2: List[Double]): Double = {
      @tailrec
      def asProd(v: List[Double], idx: Int, res: Double): Double = {
        v match {
          case Nil => res
          case ::(h, t) => asProd(t, idx + 1, res + s1(idx) * h)
        }
      }

      asProd(s2, 0, 0.0)
    }

    override def elemProd(s1: Array[Double], s2: List[Double]): List[Double] = {
      @tailrec
      def asProd(v: List[Double], idx: Int, res: List[Double]): List[Double] = {
        v match {
          case Nil => res.reverse
          case ::(h, t) => asProd(t, idx + 1, (h * s1(idx)) :: res)
        }
      }

      asProd(s2, 0, Nil)
    }

    override def elemMinus(s1: Array[Double], s2: List[Double]): List[Double] = {
      @tailrec
      def asMinus(v: List[Double], idx: Int, res: List[Double]): List[Double] = {
        v match {
          case Nil => res.reverse
          case ::(h, t) => asMinus(t, idx + 1, (h - s1(idx)) :: res)
        }
      }

      asMinus(s2, 0, Nil)
    }

  }

}
