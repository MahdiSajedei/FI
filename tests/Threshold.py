import tensorflow as tf
#import model1
import model2
#import TensorFI as ti
#import model

def Threshold(act_max):
    #Algorithm 1: Threshold Fine-Tunning
    #BEGIN ALGORITHM

    #predefined values 
    #counter >= M (M < N)
    N = 10
    sigma = 0.9  #predefined limit
    M = 9
    #act_max = 0.4  #act_max -> maximum value of activation function  
    #act_max = 10   #layer2  
    #act_max = 4.4   #layer3  
    #act_max = 11.2   #layer4  
    counter = 1  #iteration
    i = 1
    T1,T2,T3,T4=0,0,0,0
    while counter <= N :
          print("iteration:",counter)
          if i == 1 :
             S = [0,act_max]     #S -> Search distance  
             T1,T2,T3,T4,auc1_T1,auc2_T2,auc3_T3,auc4_T4 = auc_calculation(S)   #auc -> area under curve   
             i+=1 
          else :
             S,T = interval_search(T1,T2,T3,T4,auc1_T1,auc2_T2,auc3_T3,auc4_T4)    #T -> clipping Threshold
             T1,T2,T3,T4,auc1_T1,auc2_T2,auc3_T3,auc4_T4= auc_calculation(S)
             print("S:",S,"T:",T)
          #auc=[auc1_T1,auc2_T2,auc3_T3,auc4_T4]
          
          counter += 1
          for i in range(1,4):
              #delta = []
              #delta[i] = abs(auc[i+1]-auc[i])          #maximum difference between the adjacent AUCi_Ti
              if i==1:
                 delta1 = abs(auc2_T2 - auc1_T1)
                 #print(delta1)
              elif i==2:
                 delta2 = abs(auc3_T3 - auc2_T2)
                 #print(delta2)
              elif i==3:
                 delta3 = abs(auc4_T4 - auc3_T3)
                 #print(delta3)
          print("delta1:",delta1,"delta2:",delta2,"delta3:",delta3)
          if max(delta1,delta2,delta3) <= sigma and counter >= M :
             #print(T)
             return T

#END ALGORITHM

def interval_search(T1,T2,T3,T4,auc1_T1,auc2_T2,auc3_T3,auc4_T4):
    T=[T1,T2,T3,T4]
    auc=[auc1_T1,auc2_T2,auc3_T3,auc4_T4]
    #maxpos = a.index(max(a))
    index = auc.index(max(auc)) #index of T with the highest AUC
    #index = index + 1          
    print("index",index)
    if index == 3:
       Sb = [T3,T4]    #Sb -> sub-intervals
    elif index == 0 :
       Sb = [T1,T2]
    elif index == 1 :
       #Sb = [T[index-1],T[index+1]]
       Sb = [T1,T3]
    elif index == 2:
       Sb = [T2,T4]
    T = T[index]
    return Sb , T

def auc_calculation(S):
    T1 = min(S)
    T2 = T1 + (max(S)- min(S))/3
    T3 = T2 + (max(S)- min(S))/3
    T4 = max(S)
    T=[T1,T2,T3,T4]
    for i in range(4):
        with open("tmp","w") as f:
             f.write(str(T[i]))
        #to file=T[i]
        #evaluate=model1.model(T[i])   #Evaluating model using Ti ;
        _ , evaluate= model2.AlexNet8(T[i])
        #_ , evaluate = model.lenet(T[i])
        #print(evaluate)
        if i==0:
           auc1_T1= evaluate
        elif i==1:
           auc2_T2= evaluate
        elif i==2:
           auc3_T3= evaluate
        elif i==3:
           auc4_T4= evaluate

        #auc[i]= evaluate
        #Calculating AUCi ; use Trapezoidal Rule
      
    print(T1,T2,T3,T4,auc1_T1,auc2_T2,auc3_T3,auc4_T4)
    return T1,T2,T3,T4,auc1_T1,auc2_T2,auc3_T3,auc4_T4
