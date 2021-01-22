package com.eightbit85.monnn

import cate.ErrorState
import helpers.Util._

import cats.data.Reader

object Main extends App {

  val path = args(0)

  val program: Reader[Config, Either[Exception, Result]] = Reader(conf => {
    val subRout: ErrorState[List[ThetaLayer], List[Double]] = for {
      start <- ProcessingActions.startingPoint(conf)
      //gc <- ProcessingActions.gradientCheck(conf)
      fin <- ProcessingActions.learn(conf)
      //tst <- ProcessingActions.compare(conf)
    } yield List(start, fin)

    getThetas(s"$path/752.thetas").flatMap({ case (thetas, cst) =>
      cst.foreach(c => impureLog(s"Cost read from file: $c"))
      impureLog("starting...")
      subRout.run(thetas).map(Result.tupled)
    })

  }) // end: Program

  val config = Config(-0.05, 0.00001, 0.0001, 1, getFeatures(s"$path/training", 60000), 60000)
//  val config = Config(-0.05, 0.00001, 0.0001, 1, getFeatures(s"$path/testing", 10000), 10000)

  program.run(config) match {
    case Left(e) => throw e
    case Right(result) =>
      impureLog("--------------------")
      impureLog("Start/End Costs:")
      result.cost.foreach(c => impureLog(s"Cost: $c"))

      impureWrite(result, s"$path/data.thetas")
      impureLog("Done.")
  }

}