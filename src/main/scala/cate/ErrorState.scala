package com.eightbit85.monnn
package cate

import cats.Monad
import cats.implicits._
import scala.annotation.tailrec

case class ErrorState[S, A](run: S => Either[Exception,(S,A)])
object ErrorState {

  implicit def errorStateMonad[S]: Monad[({type L[a] = ErrorState[S, a]})#L] =
    new Monad[({type L[a] = ErrorState[S, a]})#L] {
      override def flatMap[A, B](fa: ErrorState[S, A])(f: A => ErrorState[S, B]): ErrorState[S, B] = {
        ErrorState(s1 => {
          fa.run(s1).flatMap({
            case (s2, a) => f(a).run(s2)
          })
        })
      }

      override def tailRecM[A, B](a: A)(f: A => ErrorState[S, Either[A, B]]): ErrorState[S, B] = {
        ErrorState(s => subby(a)(f)(s))
      }

      @tailrec
      private def subby[S, A, B](a: A)(f: A => ErrorState[S, Either[A, B]])(s: S): Either[Exception, (S, B)] = {
        f(a).run(s) match {
          case Left(e) => Left(e)
          case Right((s, Left(a1))) => subby(a1)(f)(s)
          case Right((s, Right(b))) => Right((s, b))
        }
      }

      override def pure[A](x: A): ErrorState[S, A] =  ErrorState(s => Monad[({type L[a] = Either[Exception, a]})#L].pure((s, x)))
    }

  implicit class ErrorStateOps[S, A](value: ErrorState[S, A]) {
    private val mon = Monad[({type L[a] = ErrorState[S, a]})#L]
    def map[B](f: A => B): ErrorState[S, B] = mon.map(value)(f)
    def flatMap[B](f: A => ErrorState[S, B]): ErrorState[S, B] = mon.flatMap(value)(f)
  }

}