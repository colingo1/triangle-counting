// https://spark.apache.org/docs/latest/graphx-programming-guide.html#triangle-counting


import org.apache.spark.graphx.{GraphLoader, PartitionStrategy}

// Load the edges in canonical order and partition the graph for triangle count
// val graph = GraphLoader.edgeListFile(sc, "data/graphx/followers.txt", true)
val graph = GraphLoader.edgeListFile(sc, "facebook_combined.txt", true).partitionBy(PartitionStrategy.RandomVertexCut)
// Find the triangle count for each vertex
val triCounts = graph.triangleCount().vertices

// Print the result
println(triCounts.collect().mkString("\n"))