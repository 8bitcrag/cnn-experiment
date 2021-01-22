package com.eightbit85.monnn
package helpers

import MatrixHelpers.{buildMegaSequence, mtx}
import com.eightbit85.monnn.{Convolutional, Normal, Result, ThetaLayer, TrainingPoint}
import cats.instances.double._

import java.io._
import java.nio.charset.StandardCharsets

object Util {

  def impureLog(str: String) = println(str)

  def impureWrite(finRes: Result, path: String) {
    val dataOutputStream = new DataOutputStream(new FileOutputStream(new File(path)))
    dataOutputStream.writeDouble(finRes.cost.last)
    finRes.thetas.foreach(th => {
      th.thetas.foreach(rw => {
        rw.foreach(dataOutputStream.writeDouble)
      })
    })

    dataOutputStream.close()
  }

  //TODO: remove when complete
  def time[R](name: String, block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    val diff = (t1 - t0) / 1000000
    impureLog(name + " - Elapsed time: " + diff + "ms")
    result
  }

  def useIterator[A](filename: String)(use: Iterator[Int] => A): A = {
    val bis = new BufferedInputStream(new FileInputStream(filename))
    val it: Iterator[Int] = Iterator.continually(bis.read()).takeWhile(_ > -1)
    val res = use(it)
    bis.close()
    res
  }

  def newThetas(): Either[Exception, (List[ThetaLayer], Option[Double])] = {
    val generateTheta: Double => () => Double = ep => () => (2 * ep) * math.random() - ep
    val t1 = Convolutional(mtx(5, 26, generateTheta(0.1)), buildMegaSequence(29, 0, 0, -1, 13, Nil), 1, 169)
    val t2 = Convolutional(mtx(50, 126, generateTheta(0.1)), buildMegaSequence(13, 0, 0, -1, 5, Nil), 5, 25)
    val t3 = Normal(mtx(100, 1251, generateTheta(0.1)))
    val t4 = Normal(mtx(10, 101, generateTheta(0.1)))
    Right((List(t1, t2, t3, t4), None))
  }

  def getThetas(path: String): Either[Exception, (List[ThetaLayer], Option[Double])] = {
    Right(new File(path))
      .flatMap(f => if (f.exists()) Right(f) else Left(new FileNotFoundException(s"$path does not exist")))
      .map(f => {
        val dataOutputStream = new DataInputStream(new FileInputStream(new File(path)))

        val cst = dataOutputStream.readDouble()

        val t1: List[List[Double]] = (0 until 5).map(r => (0 until 26).map(_ => dataOutputStream.readDouble()).toList).toList
        val t3: List[List[Double]] = (0 until 50).map(r => (0 until 126).map(_ => dataOutputStream.readDouble()).toList).toList
        val t4: List[List[Double]] = (0 until 100).map(r => (0 until 1251).map(_ => dataOutputStream.readDouble()).toList).toList
        val t5: List[List[Double]] = (0 until 10).map(r => (0 until 101).map(_ => dataOutputStream.readDouble()).toList).toList

        val mseq1 = buildMegaSequence(29, 0, 0, -1, 13, Nil)
        val mseq2 = buildMegaSequence(13, 0, 0, -1, 5, Nil)
        val the1 = Convolutional(t1, mseq1, 1, 169)
        val the2 = Convolutional(t3, mseq2, 5, 25)
        val the3 = Normal(t4)
        val the4 = Normal(t5)

        dataOutputStream.close()
        (List(the1, the2, the3, the4), Some(cst))
      })
  }

  def thetasToJson(thetas: List[ThetaLayer], path: String) = {
    val dataOutputStream = new FileWriter(path, StandardCharsets.UTF_8)
    dataOutputStream.write("[")
    thetas.foreach(th => {
      dataOutputStream.write("[")
      th.thetas.foreach(rw => {
        dataOutputStream.write("[")
        rw.foreach(t => dataOutputStream.write(t.toString + ","))
        dataOutputStream.write("],")
      })
      dataOutputStream.write("],")
    })
    dataOutputStream.write("]")
    dataOutputStream.close()
  }

  def getFeatures(path: String, amnt: Int): List[TrainingPoint] =
    useIterator(s"$path-data") { it =>
      useIterator(s"$path-labels") { la =>
        la.drop(8)
        val meta: Seq[Int] = it.take(16).grouped(4).toVector.map(g => {
          g.reduce((a, b) => (a << 8) ^ b)
        })

        val rows: Int = meta(2)
        val cols: Int = meta(3)
        val pixels: Int = rows * cols

        it.grouped(pixels).take(amnt).map(g => {
          val y = la.next()
          val r = (0 to 9).map(i => if (i == y) 1.0 else 0.0).toList
          val padded = List.fill(29)(0) ::: g.grouped(28).flatMap(rw => rw.appended(0)).toList
          val mx = padded.max.toDouble
          val d = padded.map(_ / mx)
          TrainingPoint(List(d), r)
        }).toList
      }
    }

}
