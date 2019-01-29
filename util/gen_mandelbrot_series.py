#!/usr/bin/env python3

DEGREE = 32

print("T(2)*S[0]*p + T(1),")

for i in range(1, DEGREE):
    print("T(2)*(S[%d]*p" % i, end='')

    for j in range(0, i//2):
        print(" + S[%d]*S[%d]" % (j, i-j-1), end='')

    print((") + S[%d].sqr()," % (i//2)) if i%2 == 1 else "),")
