from math import *
import matplotlib.pyplot as plt
n=4
G=(1+5**.5)/2
print('Golden ratio =', G, 'Number of quarters =' , n)
w=250
print('Width =', w)
k=pi/180
pt =[(G**(j/90)*cos(j*k)+w/2,G**(j/90)*sin(j*k)+w/2)for j in range(n*90)]
#print(pt) # the (x,y) tuple
plt.scatter (*zip(*pt)) # plot the tuples
plt.show()


