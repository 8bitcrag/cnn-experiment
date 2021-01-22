package com.eightbit85.monnn
package helpers

import cats.Monoid

import scala.annotation.tailrec

object MatrixHelpers {

  def mtx[A: Monoid](rows: Int, cols: Int, generator: () => A): List[List[A]] = {
    (1 to rows).map(_ => {
      (1 to cols).map(_ => generator()).toList
    }).toList
  }

  @tailrec
  private def generateTranspose[A: Monoid](trans: List[List[A]], mtx: List[List[A]]): List[List[A]] = {
    val updated = (for (row <- mtx) yield row.head) :: trans
    val reduced = for (row <- mtx) yield row.tail
    reduced.head match {
      case Nil => updated.reverse
      case _ => generateTranspose(updated, reduced)
    }
  }

  def transpose[A: Monoid](mtx: List[List[A]]): List[List[A]] = generateTranspose(List.empty, mtx)

  val sigmoid: Double => Double = x => 1 / (1 + math.exp(-x))
  val sigmoidDiff: Double => Double = sig => sig * (1 - sig)

  val gradientsToSizeCheck: Double => List[List[List[Double]]] => Boolean =
    epsilon => grads => grads.exists(m => m.exists(v => v.exists(e => math.abs(e) > epsilon)))

  @tailrec
  def buildMegaSequence(inputWidth: Int, idx: Int, start: Int, end: Int, max: Int, tgt: List[List[Int]]): List[List[Int]] = {
    val st = if (idx % 2 != 0 && idx >= 5) start + 1 else start
    val en = if (idx % 2 == 0 && end < max - 1) end + 1 else end
    if (st == max) return tgt.reverse
    val s: List[List[Int]] = (st to en).map(_ * max).toList.map(i => buildSequence(0, i, i - 1, i + max, Nil))
      .reduceLeft((a, c) => a.zip(c).map(t => t._1 ::: t._2))
    buildMegaSequence(inputWidth, idx + 1, st, en, max, s ::: tgt)
  }

  @tailrec
  def buildSequence(idx: Int, start: Int, end: Int, max: Int, tgt: List[List[Int]]): List[List[Int]] = {
    val st = if (idx % 2 != 0 && idx >= 5) start + 1 else start
    val en = if (idx % 2 == 0 && end < max - 1) end + 1 else end
    if (st == max) return tgt
    buildSequence(idx + 1, st, en, max, (st to en).toList :: tgt)
  }

}
