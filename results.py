import os
import numpy as np
import matplotlib.pyplot as plt

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
        for line in fr:
            elements = line.split(" ")
            if elements[0] == "mean" and elements[1] == "average" and elements[2] == "precision:":
                ap = elements[3]
        return float(ap)

def get_average_roc(filename):
    with open(filename, 'r', encoding="utf8") as fr:
        fr.readline()
        roc = []
        for line in fr:
            elements = line.split(" ")
            if elements[0] == "ROC":
                roc.append(float(elements[5]))

        return np.mean(roc)

def elapsed_time_mAP_graph(time, mAP):
    plt.figure()
    plt.xlabel('elapsed time')
    plt.ylabel('mAP')
    plt.plot(time, mAP)
    plt.title('Variation of mAP according to inference duration')
    plt.savefig('./other/graphs/mAP-time')


def elapsed_time_roc_graph(time, roc):
    plt.figure()
    plt.xlabel('elapsed time')
    plt.ylabel('mean roc auc')
    plt.plot(time, roc)
    plt.title('Variation of mean roc auc according to inference duration')
    plt.savefig('./other/graphs/rocauc-time')


if __name__ == '__main__':
    model_names = ["size100", "size200", "size300"]
    time = []
    mAP = []
    roc = []
    for name in model_names:
        RPI_logs = "./output_files/logs_" + name + ".txt"
        metrics = "./output_files/metrics_" + name + ".txt"
        _, process_time = get_average_process_time(RPI_logs)
        time.append(round(process_time, 3))
        roc.append(round(get_average_roc(metrics), 3))
        mAP.append(round(get_average_precision(metrics), 3))

    elapsed_time_roc_graph(time, roc)
    elapsed_time_mAP_graph(time, mAP)
