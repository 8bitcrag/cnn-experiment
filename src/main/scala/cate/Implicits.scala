package com.eightbit85.monnn
package cate

import cats.Monoid
import cats.syntax.semigroup._

object Implicits {

  implicit def ElemWiseListMonoid[A: Monoid]: Monoid[List[A]] = new Monoid[List[A]] {
    /**
     * This is not technically a correct unit value but there is a reason for it.
     * Consider the monoid of all ordered sets of integers of size 3. The
     * unit element would be [0, 0, 0]. The combine morphism represents this with
     * zipAll. So although not technically correct, this means not having to write
     * an infinite number of monoids and conceptually represents the right thing.
     */
    override def empty: List[A] = List.empty

    override def combine(x: List[A], y: List[A]): List[A] = {
      x.zipAll(y, Monoid[A].empty, Monoid[A].empty)
        .map(t => t._1 |+| t._2)
    }
  }

}
