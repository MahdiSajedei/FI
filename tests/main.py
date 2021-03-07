import tensorflow as tf
import os
import sys
import Threshold as T
#import model1
#import model
import model2
#import numpy as np
#l1=6.478024691358024
#l2=3.8443072702331964
#l3=6.28641975308642
#l4=7.418353909465021
def main():

    enable = 1
    if enable == 1:
       act_max = 10
       #find best Threshold
       TT=T.Threshold(act_max)
       log = open("Threshold.txt", "w")
       log.write(str(TT))
       print(TT)
    else :
       #mean accuracy of n run model
       acc = open("mean_accuracy.txt", "w")
       result = 0
       n = 50
       for i in range(n):
           #res = model1.model(i)
           #res , _ = model.lenet(i)
           res , _ = model2.AlexNet8(i)
           result += res 
       result = result / n
       acc.write(str(result))
    
if __name__ == '__main__':
    main()

