from Algorithm import Algorithm, HEURISTICS
import argparse
import os
import logging
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Illustration of A* algorithm")
    parser.add_argument("--heuristic", "--he", type=str, help="Heuristic choice", required=False,
                        choices=HEURISTICS, default="Mean")
    parser.add_argument("--instance", type=str, help="Path to instance",
                        required=False, default=Path("datasets/20_nodes.txt"))
    parser.add_argument("-b", "--bidirect", action='store_true', help="bidirectionnal", required=False)
    parser.add_argument("--log", dest="logLevel", default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logger level")
    args = parser.parse_args()

    logging.basicConfig(format='[%(levelname)s] : %(message)s')
    logger = logging.getLogger('shortest-path')
    logger.setLevel(args.logLevel)

    logger.debug(args)
    instance = args.instance
    if not os.path.isfile(instance):
        logger.warning("Instance \"{}\" not found".format(instance))
        exit(1)

    algorithm = Algorithm(instance, heuristic=args.heuristic,
                          logger=logger, is_bidirectional=args.bidirect)
    algorithm.run()
    if(len(algorithm.G.nodes) > 50):
        answer = input("Many nodes to draw ({}), confirm drawing? [y/N]\n".format(len(algorithm.G.nodes)))
        if(answer != "y"):
            return
    algorithm.show()


if __name__ == "__main__":
    main()
