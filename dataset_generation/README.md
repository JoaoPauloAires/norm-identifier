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

- problem path: Path to the generated problem [use toy_problem](problems/toy-problem.prblm)

4 - Check the [observers](observers/) folder to obtain the dataset for each observer. 

### 1 - Generate a Problem

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

- p and r: We represent these two properties the same way. 'p' describes forbidden nodes while 'r' describes the presence of traffic lights with red signs. Both elements are followed by a list of node numbers separated by dashes indicating the nodes affected by them. The following examples describes these properties.

    ```p 0-4-2```

    ```r 1-4```

- car: This property describes cars in the graph. We represent them by a sequence of values separated by dashes. The first value indicates the initial position (node) of the car. The second value indicates its goal position (node). The third value indicates the initial speed of the car. Finally, the fourth value indicates the probability of complying with a node speed limit. We will probably remove this value as we aim to improve car's ability to plan. An example of the car line can be found below. This line repeats as much as the number of cars in the graph.

    ```car 1-3-2-0.5```

- sp: This property describes the speed limits in nodes. We divide speed limits by commas and each description has a speed limit followed by a sequence of nodes that has its limitation. The following example illustrates the line. In this example, there are two speed limits (3 and 4), nodes 0 and 3 have a speed limit of 3, and nodes 1, 2, and 4 have a speed limit of 4. In this work, we consider speed as an abstract unit that goes, usually, from 0 to 10.

    ```sp 3-0-3,4-1-2-4```

- ob and enf: Both ob and enf have the same pattern of representation. 'ob' stands for observers and 'enf' stands for enforcers. Each line with represents one agent of the type, we build it by node numbers separated by dashes. The nodes represent the view range of them. The following examples illustrate these lines. Observers and enforcers can monitor the same nodes.

    ```ob 1-4```

    ```enf 1-3```