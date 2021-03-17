from multiprocessing import Process
import time

def f1(name):
    time.sleep(3)
    print('hello', name)

def f2(name):
    print('world')

if __name__ == '__main__':

    p2 = Process(target=f2, args=('bob',))
    p1 = Process(target=f1, args=('bob',))
    # p2.start()
    flag = True
    while True:
        if flag:
            p1.start()
            flag = False
        if p1.is_alive():
            print("A")
        else:
            print("D")
