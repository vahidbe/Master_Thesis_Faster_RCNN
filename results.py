import os
import numpy as np

def get_average_process_time(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        elapsed_time = []
        for line in fr:
            elements = line.split(" ")
            if elements[0] != "Elapsed":
                continue

            elapsed_time.append(float(elements[3]))

        return len(elapsed_time), np.mean(elapsed_time)

def get_average_precision(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        ap =
        for line in fr:
            elements = line.split(" ")
            if elements[0] == "mean" and elements[1] == "average" and elements[2] == "precision:":
                ap = elements[3]

        return ap

def get_average_roc(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        roc = []
        for line in fr:
            elements = line.split(" ")
            if elements[0] == "ROC":
                roc.append(float(elements[5]))

        return np.mean(roc)

def elapsed_time_accuracy_graph():
    pass

if __name__ == '__main__':
    res = get_average_process_time("logs.txt")
    print(res)