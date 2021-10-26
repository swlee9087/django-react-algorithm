from django.test import TestCase

# Create your tests here.
from admin.nlp.models import NaverMovie

if __name__ == '__main__':



    # basic list compre:
    # [( ) for ( ) in [] ]
    # [i for i in []]
    # [(i,j) for [i,j] in []]
    dc1 = {}
    dc2 = {}
    dc3 = {}  # dict-ize
    ls1 = ['10', '20', '30', '40', '50']
    ls2 = [10, 20, 30, 40, 50]

    # method 1 - range()
    # i for i in range(0, len(ls1))
    # for i in range(0, len(ls1)):
    #     dc1[ls1[i]] = ls2[i]
    # dc1 = [i for i in range(0,len(ls1)) if dc1[ls1[i]] = ls2[i]]  NOPE
    # [dc1[ls1[i]: ls2[i]: for i in range(0,len(ls1)]
    # [ls2[i]: ls1[i] for i in range(0, len(ls1))]
    # dc1 = {ls2[i]: ls1[i] for i in range(0, len(ls1))}
    dc1 = {ls1[i]: ls2[i] for i in range(0, len(ls1))}

    # method 2 - zip()
    # (i, j) for i, j in []
    # for i, j in zip(ls1, ls2):
    #     dc2[i] = j
    # dc2 = {j: i for i in zip(ls1) for j in zip(ls2)}  NOT REALLY
    dc2 = {i: j for i, j in zip(ls1, ls2)}

    # method 3 - enumerate()
    # for i, j in enumerate(ls1):
    #     dc3[j] = ls2[i]
    # dc3 = {j: j for i in enumerate(ls1) for j in ls2[i]}  NAH
    dc3 = {j: ls2[i] for i, j in enumerate(ls1)}

    print('*~*' * 20)
    print(dc1)
    print('~*~' * 20)
    print(dc2)
    print('*==' * 20)
    print(dc3)


