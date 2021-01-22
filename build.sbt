name := "cnn-experiment"

version := "0.1"

scalaVersion := "2.13.4"

idePackagePrefix := Some("com.eightbit85.monnn")

libraryDependencies ++= Seq(
  "org.typelevel" %% "cats-core" % "2.1.1",
  "org.typelevel" %% "cats-free" % "2.1.1",
  "org.typelevel" %% "discipline-munit" % "0.3.0",
  "org.scalameta" %% "munit" % "0.7.14" % Test,
  "org.typelevel" %% "cats-laws" % "2.0.0" % Test,
  "com.github.alexarchambault" %% "scalacheck-shapeless_1.14" % "1.2.3" % Test,
  "org.scalameta" %% "munit-scalacheck" % "0.7.14" % Test,
  "org.scalactic" %% "scalactic" % "3.2.0"
)

testFrameworks += new TestFramework("munit.Framework")
Test / parallelExecution := false