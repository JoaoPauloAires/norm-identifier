# Violation Identifier

In this repository, we build a violation identifier.
We divide this project into two sections: dataset generation and violation identificaction.


### Dataset Generation

In this section, we created scripts to make a dataset from a road-like graph structure.
The graph consists of parts of a road (nodes) and its conections (arcs).
Along with the graph, we estipulate a car position (one of the nodes).
Thus, we create a series of states (graph plus car positions), which we consider a plan.
Besides, we define a set of norms over the graph, e.g., a forbidden node and a speed limit.
