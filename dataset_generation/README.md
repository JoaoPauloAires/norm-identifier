# Dataset Generation

In this folder, we organize our scripts of dataset generation.
The process of generating a dataset consists of three steps:

### TL;DR

1 - Set the parameters in [problem_gen_dict.json](problem_gen_dict.json).

2 - Execute [problem_generator.py](problem_generator.py):

    ```python problem_generator.py max_nodes config_path versions```

- max_nodes: maximum number allowed to generate the graph.

- config_path: path to .json file containing the setup parameters to create the problem.

- versions: number of problems to create.

3 - Execute [graph_gen.py](graph_gen.py):

    ```python graph_gen.py problem_path```

- problem path: Path to the generated problem. To test, [use toy_problem](problems/toy-problem.prblm).

4 - Check the [observers](observers/) folder to obtain the dataset for each observer. 

### 1 - Generate a Problem


##### 1.1 - Problem setting

In order to generate a planning problem that will result in a graph, we use a setting file.
See [toy-problem](problems/toy-problem.prblm) to understand our setting file.
The file has a coding scheme that we parse to generate the graph.
Thus, each line in the file has an identifier that indicates a graph property.
We defined eight different properties:

- n: Describes the number of nodes in the graph.
It is followed by a sequence of numbers that go from 0 to the desired number of nodes - 1.
The following example describes a graph with 5 nodes.

    ```n 0,1,2,3,4```


- c: Describes the connections between nodes and their probabilities.
Each element of this property has three components, the first and second indicate the connection (from-to) between two node numbers (previously described in 'n').
The third component is a float number indicating the probability of using the connection.
The following example demonstrates a series of connections.

    ```c 0-0-0.4,0-4-0.6,1-3-1.0,2-1-1.0,4-3-1.0```

- p and r: We represent these two properties in the same way.
'p' describes forbidden nodes while 'r' describes the presence of traffic lights with red signs.
Both elements are followed by a list of node numbers separated by dashes indicating the nodes affected by them.
The following examples describes these properties.

    ```p 0-4-2```

    ```r 1-4```

- car: This property describes cars in the graph. We represent them by a sequence of values separated by dashes.
The first value indicates the initial position (node) of the car.
The second value indicates its goal position (node).
The third value indicates the initial speed of the car.
Finally, the fourth value indicates the probability of complying with a node speed limit.
We will probably remove this value as we aim to improve car's ability to plan.
An example of the car line can be found below.
This line repeats as much as the number of cars in the graph.

    ```car 1-3-2-0.5```

- sp: This property describes the speed limits in nodes.
We divide speed limits by commas and each description has a speed limit followed by a sequence of nodes that has its limitation.
The following example illustrates the line.
In this example, there are two speed limits (3 and 4), nodes 0 and 3 have a speed limit of 3, and nodes 1, 2, and 4 have a speed limit of 4.
In this work, we consider speed as an abstract unit that goes, normally, from 0 to 10.

    ```sp 3-0-3,4-1-2-4```

- ob and enf: Both ob and enf have the same pattern of representation.
'ob' stands for observers and 'enf' stands for enforcers.
Each line represents one agent of the type, we represent it by node numbers separated by dashes.
The nodes express the view range of them.
The following examples illustrate these lines.
Observers and enforcers can monitor the same nodes.

    ```ob 1-4```

    ```enf 1-3```

##### 1.2 - Problem Generator

Now that we understand how a problem setting file works, we can generate automatically generate it using the [problem_generator.py](problem_generator.py) script.
This script requires a single parameter that defines some limitations of what will be generated.
We set these parameters in a (json file)[problem_gen_dict.json].
It contains 10 parameters that we use as basis to create the graph.
We use float values in this parameters as a percentage related to the number of nodes.
The entire process relies on random selections, thus, the result of the number of nodes regulates the number of the other parameters, such as connections, prohibition nodes, speed, among other.

- min_nodes: int value that states a minimum limit to the number of nodes in the graph;
- max_connections: float number that, based on the number of nodes, will set the max # of connections in the graph;
- max_prohibitions: float value to limit the max number of forbidden nodes in graph;
- max_red_signals: float value to limit the max number of red signs in nodes with traffic lights;
- max_speed: int number defining the max speed a car can have;
- min_speed: int number defining the min speed a car can have;
- speed_limits: float value to limit the max number of speed limits in graph;
- max_cars: float value to limit the max number of cars in graph;
- max_obs: float value to limit the max number of observers in graph;
- max_enfs: float value to limit the max number of enforcers in graph;
- max_range: float value to limit the max number of nodes enforcers and observers can monitor at the same time.

To execute problem_generator.py use the following command:

```python problem_generator.py max_nodes config_path versions```

- max_nodes: maximum number of nodes to generate the graph.

- config_path: path to a json file containing the setup parameters to create the problem.

- versions: number of problems to create.

### 2 - Generate Graph and Observer Datasets

After generating the problem (check [toy-problem.prblm](problems/toy-problem.prblm)), we use it to create the graph, run the plans of car agents, and obtain the observers dataset.
The [graph_gen.py](graph_gen.py) script takes a single parameter, which is the problem path.
To execute the script, use the following command:

```python graph_gen.py problem_path```

- problem path: Path to the generated problem. To test, [use toy_problem](problems/toy-problem.prblm).