# INFO-H410 Project

Project conducted by Alexandre MISSENARD, Andrey SOBOLEVSKY and Franck TROUILLEZ.

This project has the aim to implement an A* algorithm for the shortest path problem in a nodes graph. The A* algorithm is implemented in `Algorithm`. 

3 sub-classes are created to define the heuristics. 

The Dijkstra's algorithm is also implemented, in order to compare with A*. It is done in `Dijkstra`. 

The instances can be found in the directory `datasets`. In order to read these instances, a class `FileHandler` is created, and allows to read the info in the file. In order to run the algorithms, simply run the `main` file. 

The algorithm class `Algorithm` can also use a bidirectional search, instead of the classical unidirectional search.

In order to generate the instances, an instance generator has been created, which can be found in `InstanceGenerator`.

