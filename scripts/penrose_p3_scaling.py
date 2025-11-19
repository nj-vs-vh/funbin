import math

from funbin.penrose import penrose_P3_de_Brujin

for n in range(3, 50):
    polys = penrose_P3_de_Brujin(n, 2 * n)
    print(n, len(polys), math.sqrt(len(polys) / 2))
