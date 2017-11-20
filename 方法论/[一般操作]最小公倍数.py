'''最小公倍数操作
   2017.11.20

'''
from functools import reduce
def gcd(a, b):
    r = a % b
    if r:
        return gcd(b, r)
    else:
        return b
#print gcd(13, 6)

def lcm(a, b):
    return a * b / gcd(a, b)
#print lcm(12, 6)

def lcmAll(seq):
    if len(seq)==1:
        return seq[0]
    return reduce(lcm, seq)

#lis = [a for a in range(1, 11)]
#print lcmAll(lis)