# Norm Identifier

This repository contains a norm identifier approach to operate over multi-agent systems.
Our approach consists of training a model to learn violation states.
We propose a traffic environment where observer agents try to learn traffic norms by observing cars moving through the streets.
We divide this project into two sections: dataset generation and norm identification.


### Dataset Generation

In this section, we have scripts to make a dataset from a road-like graph structure.
The graph consists of streets (nodes) and their connections (arcs).
Nodes in the graph may have three properties: traffic lights; prohibition; speed limit.

- Traffic lights can be either red or green;
- Prohibition refers to forbidden nodes that cars must not enter;
- Speed limit regulates the internal speed of nodes.

### Models

In this section, we have learning algorithms to learn norms.
We developed two models, the first one is an SVM and the second a GRU.
SVM models can deal with high-dimension problems.
GRU models can preserve temporal information.
Both approaches can have benefits on our scenario that involves the use of samples with big dimensionality and sensitive to temporal information.
