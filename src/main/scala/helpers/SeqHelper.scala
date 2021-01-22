package com.eightbit85.monnn
package helpers

import MatrixHelpers.sigmoid

import scala.collection.Seq

trait SeqHelper[F[Double]] {

  def elemMinus(s1: F[Double], s2: List[Double]): List[Double]

  def elemProd(s1: F[Double], s2: List[Double]): List[Double]

  def elemProdSum(s1: F[Double], s2: List[Double]): Double

  val mtxVecToProduct: Seq[F[Double]] => List[Double] => List[Double] =
    m => v => (for (row <- m) yield elemProdSum(row, v)).toList

  val toActivations: Seq[F[Double]] => List[Double] => List[Double] =
    t => i => for (el <- mtxVecToProduct(t)(1 :: i)) yield sigmoid(el)

}

object SeqHelper {
  def apply[F[_]](implicit instance: SeqHelper[F]): SeqHelper[F] = instance
}

object SeqHelperSyntax {

  implicit class SeqHelperOps[F[Double]](value: F[Double]) {
    def elemProdSum(l: List[Double])(implicit helper: SeqHelper[F]): Double =
      helper.elemProdSum(value, l)

    def elemProd(l: List[Double])(implicit helper: SeqHelper[F]): List[Double] =
      helper.elemProd(value, l)

  }

}