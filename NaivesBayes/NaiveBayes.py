# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:39:09 2018

@author: anurag
"""

from math import log 
import numpy as np
import csv
#file names
##############################################
train_data_file = "train_data.csv"            #train data
train_data_label = "train_label.csv"          #train label
vocabulary_file = "vocabulary.txt"            #vocabulary
test_data_file = "test_data.csv"              #test data
test_data_label = "test_label.csv"            #test label
##############################################

#method to get labels of each document
##############################################
def get_labels(filename):
    with open(filename,'r') as f:
        c=1
        for line in csv.reader(f):
            cl = int(line[0])
            data_labels[c]=cl
            c+=1;


#Definition for calculating total_documents
##############################################
def get_last_row(csv_filename):                      # getting the last line of train_data
    with open(csv_filename, 'r') as f:               #.
        lastrow = None                               #.
        for lastrow in csv.reader(f): pass           # will stop at the last row
        return lastrow           
##############################################

#Calculating total number of documents in each class
##############################################
def get_documents_in_each_category(trainlabel_file):
    with open(trainlabel_file,'r') as f:
        row = None
        for row in csv.reader(f):
            i = int(row[0])
            documents_in_class[i] += 1
##############################################   

#Calculating total number of words in each class
##############################################
def get_words_in_each_class(traindata_file):     
    c = 1
    with open(traindata_file,'r') as f:
      row = None
     
      for row in csv.reader(f):
            cl = int (row[0])
            j = int(row[1])
            k = int(row[2])
            
            c= data_labels[cl]
            if j in word_count_in_class[c]:
                word_count_in_class[c][j] += k
            else:
                word_count_in_class[c][j] = k
############################################## 

#get |Vocabulary|
############################################## 
def get_vocabulary_count(vacb_file):
    v_c = 0
    with open(vocabulary_file) as f:
        row = None
        for row in f:
            v_c+=1
        return v_c
##############################################    
            
            
#calculating bayesian estimate and maximum likelihood estimate
##############################################
def get_b_e(filename,vocabulary_count):
    
    with open(filename,"r") as f1:
        P_be = [0 for i in range(21)]
        P_mle =[0 for i in range(21)]
        cat_count = 1
        next_doc = 2
        for f1_row in csv.reader(f1):
            word_id = int(f1_row[1])
            doc_id = int(f1_row[0])
            
            if next_doc==doc_id:
               next_doc += 1
               
               if doc_id > category_sum_count[cat_count]:
                    cat_count += 1
               actual = data_labels[doc_id]
               for i in range(1,21):
                   P_be[i] += log(class_priors[i] )
                   P_mle[i] += log(class_priors[i] )
               
               confusion_matrix[actual][np.argmax(P_be[1:21])+1] +=1
               mle_confusion_matrix[actual][np.argmax(P_mle[1:21])+1] +=1
               P_be = [0 for i in range(21)] 
               P_mle =[0 for i in range(21)]
           
            for i in range(1,21):
              if word_id in word_count_in_class[i]:
                n_k = word_count_in_class[i][word_id] 
                P_mle[i] += log(((n_k)/(total_words_in_class[i] ))) 
              else:
                n_k = 0
              P_be[i] += log(((n_k+1)/(total_words_in_class[i]+vocabulary_count )))
                          
##############################################   


category_sum_count = [0 for i in range(21)]
documents_in_class = [0 for i in range(21)]

word_count_in_class = [dict() for x in range(21)]
total_words_in_class = [0 for i in range(21)]
confusion_matrix = [ [0] * 21 for _ in range(21)]
mle_confusion_matrix = [ [0] * 21 for _ in range(21)]
data_labels = dict()
get_labels(train_data_label)

total_documents = int(get_last_row(train_data_file)[0])  # getting the total number of documents          
vocabulary_count = get_vocabulary_count(vocabulary_file) #getting |vocabulary|

get_documents_in_each_category(train_data_label)        #getting documents in each category

#category_docs_sum
category_sum_count = documents_in_class.copy()        
for i in range(1,21):
    category_sum_count[i] += category_sum_count[i-1]

get_words_in_each_class(train_data_file)
for i in range(1,21):
    total_words_in_class[i]=sum(word_count_in_class[i].values())


#Printing trainning results
##############################################
class_priors =[0 for i in range(21)]
print ("1. Class priors")
print ("------------------------------------------------------------------------------------------------")        
for i in range(20):
    print ("P(Omega=",i+1,"):",documents_in_class[i+1]/total_documents)
    class_priors[i+1] = documents_in_class[i+1]/total_documents
##############################################
get_b_e(train_data_file,vocabulary_count)          # classifying train data using bayesian estimate

correct = 0
for i in range(1,21):
        correct += confusion_matrix[i][i]
print ("\n")
print("2.Results based on Bayesian estimator")
print("\n")
print("2.1).Training data on Bayesian estimator")
print ("Overall Accuracy=",correct/total_documents)
print ("------------------------------------------------------------------------------------------------")
print ("Class Accuracy:")  
print ("------------------------------------------------------------------------------------------------") 
for i in range(1,21):
    print ("Group",i,":",confusion_matrix[i][i]/documents_in_class[i])
print("Confusion Matrix")
print ("------------------------------------------------------------------------------------------------")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in confusion_matrix]))
print(" ")
################################################

#methods for test_data
##############################################
def test_get_b_e(filename,vocabulary_count):
     with open(filename,"r") as f1:
        P_be = [0 for i in range(21)]
        P_mle =[0 for i in range(21)]
        cat_count = 1
        next_doc = 2
        for f1_row in csv.reader(f1):
            word_id = int(f1_row[1])
            doc_id = int(f1_row[0])
            word_count = int(f1_row[2])
            if next_doc==doc_id:
               next_doc += 1
               
               if doc_id > test_category_sum_count[cat_count]:
                    cat_count += 1
               actual = test_data_labels[doc_id]
               for i in range(1,21):
                   P_be[i] += log(class_priors[i] )
                   P_mle[i] += log(class_priors[i])
               
               test_confusion_matrix[actual][np.argmax(P_be[1:21])+1] +=1
               test_mle_confusion_matrix[actual][np.argmin(P_mle[1:21])+1] +=1
               P_be = [0 for i in range(21)] 
               P_mle = [0 for i in range(21)]
               
           
            for i in range(1,21):
              if word_id in word_count_in_class[i]:
                n_k = word_count_in_class[i][word_id] 
                P_mle[i] += ((log(((n_k)/(total_words_in_class[i]))))*word_count) 
              else:
                n_k = 0
              P_be[i] += ((log(((n_k+1)/(total_words_in_class[i]+vocabulary_count ))))*(word_count))


def test_get_labels(filename):
    with open(filename,'r') as f:
        c=1
        for line in csv.reader(f):
            cl = int(line[0])
            test_data_labels[c]=cl
            c+=1;

def test_get_documents_in_each_category(testlabel_file):
    with open(testlabel_file,'r') as f:
        row = None
        for row in csv.reader(f):
            i = int(row[0])
            test_documents_in_class[i] += 1
############################################## 
            
            
#test data
##############################################            
test_category_sum_count = [0 for i in range(21)]
test_documents_in_class = [0 for i in range(21)]

test_confusion_matrix = [ [0] * 21 for _ in range(21)]     #confusion matrix for test data 
test_mle_confusion_matrix = [ [0] * 21 for _ in range(21)] #confusion matrix for test data using MLE
test_total_documents = int(get_last_row(test_data_file)[0])          
         
test_get_documents_in_each_category(test_data_label)       #test documents in each category
test_data_labels=dict()
test_get_labels(test_data_label)                           #actual labels for test documents

#category_docs_sum
test_category_sum_count = test_documents_in_class.copy()        
for i in range(1,21):
    test_category_sum_count[i] += test_category_sum_count[i-1]

test_get_b_e(test_data_file,vocabulary_count)          #classify test using bayesian estimate and mle
##############################################

#printing test data
##############################################
correct = 0
for i in range(1,21):
        correct += test_confusion_matrix[i][i]
print("2.2).Test Data on Bayesian estimator")
print ("------------------------------------------------------------------------------------------------")
print ("Overall Accuracy=",correct/test_total_documents)
print ("------------------------------------------------------------------------------------------------")
print ("Class Accuracy:")  
print ("------------------------------------------------------------------------------------------------")
for i in range(1,21):
    print ("Group",i,":",test_confusion_matrix[i][i]/test_documents_in_class[i])
print(" ")
print("Confusion Matrix")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in test_confusion_matrix])) 
print("")    
print("3.Results based on Maximum Likelihood estimator")  
print ("------------------------------------------------------------------------------------------------") 
print("2.2).Test Data on Maximum Likelihood estimator")
print ("------------------------------------------------------------------------------------------------")
correct = 0
for i in range(1,21):
        correct += test_mle_confusion_matrix[i][i]
print ("\n")
print ("Overall Accuracy=",correct/test_total_documents)
print ("------------------------------------------------------------------------------------------------")
print ("Class Accuracy:")  
print ("------------------------------------------------------------------------------------------------") 
for i in range(1,21):
    print ("Group",i,":",test_mle_confusion_matrix[i][i]/test_documents_in_class[i])
print("Confusion Matrix")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in test_mle_confusion_matrix]))

