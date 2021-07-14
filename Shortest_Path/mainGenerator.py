from InstancesGenerator import InstancesGenerator


if __name__ == "__main__":
    instance_generator = InstancesGenerator(numberOfNodes = 10000, minX = 0, minY = 0, maxX = 500, maxY = 500, probabilityOfStrangeMove = 0.00, minConnexions = 1, maxConnexions = 10, minRadius = 2, maxRadius = 30)
    instance_generator.generate("testF.txt")
