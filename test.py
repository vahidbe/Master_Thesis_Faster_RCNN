import itertools as it

if __name__ == "__main__":
    my_dict = {'A': ['D', 'E'], 'D': ['F', 'G', 'H'], 'C': ['I', 'J']}
    allNames = my_dict.keys()
    combinations = it.product(*(my_dict[Name] for Name in allNames))
    for param in combinations:
        print(param[0])