# INFO-H410 Project

Project conducted by Alexandre MISSENARD, Andrey SOBOLEVSKY and Franck TROUILLEZ.

This project has the aim to implement an A* algorithm for the shortest path problem in a nodes graph. The A* algorithm is implemented in `Algorithm`. 

3 heuristics are defined : Manhattan, euclidian and chebyshev distances.

Null heuristic is also implemented as an illustration of the Dijkstra algorithm.

The instances can be found in the directory `datasets`. 

In order to read these instances, a class `FileHandler` is created, and allows to read the info in the file.

The class `GUI` sequentially displays A* search results for illustration purpose.

The algorithm class `Algorithm` can also use a bidirectional search, instead of the classical unidirectional search.

In order to run the algorithms, simply run the `main` file. 

## Usage  
Running code:  
```
poetry run python main.py --he Euclidian --instance datasets/20_nodes.txt  
```
Detailled usage:
```
usage: main.py [-h] [--heuristic {Manhattan,Euclidian,Chebyshev,Dijkstra}] [--instance INSTANCE] [-b] [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}]

Illustration of A* algorithm

optional arguments:
  -h, --help            show this help message and exit
  --heuristic {Manhattan,Euclidian,Chebyshev,Dijkstra}, --he {Manhattan,Euclidian,Chebyshev,Dijkstra}
                        Heuristic choice
  --instance INSTANCE   Path to instance
  -b, --bidirect        bidirectionnal
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logger level
```  

Example of command:
```
poetry run python main.py --he Euclidian --instance datasets\13_nodes.txt
```
