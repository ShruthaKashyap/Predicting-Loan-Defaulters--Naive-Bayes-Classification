from __future__ import print_function

import sys
from operator import add
from pyspark import SparkContext

#For Naive Bayes
#dictionary containing number of rows for each target value
count_c = dict()
prob_c = dict()

#total rows in data
totalRows = 0


#Delimeter to separate column value from targetValue
delim = '_#_'
tdict = dict()

#Target Column heading name (Same name should be used/indexed in catCols dictionary)
pred_var = 'loan_status'

catCols = dict()
catCols['term'] = 0
catCols['grade'] = 1
catCols['emp_length']=2
catCols['home_ownership'] = 3
catCols['verification_status'] = 4
catCols['loan_status'] = 5
catCols['initial_list_status'] = 6

def processRow(row):
    tokens = row.split(',')

    target = tokens[catCols['loan_status']]
    col_list = []
    
    for col in catCols:
        col_list.append((col+'.'+tokens[catCols[col]],1))
        col_list.append((col+'.'+tokens[catCols[col]]+delim+str(target),1))
        
        
    return col_list

def findTargets(tup):
    
    if tup[0].startswith(pred_var):
        if delim in tup[0]:
            return (tup[0].split(delim)[1],tup[1])
        else:
            return ''
    else:
        return ''

def convertToProb(tupl):
    
    targ = tupl[0].split(delim)[1]
    return (tupl[0],tupl[1]/prob_c[targ])

def sortAndGrpTuples(lst):
    mydict = dict()

    for i in lst:
        tdata = i[0].split('.')
        if  tdata[0] not in mydict:
            mydict[tdata[0]] = dict()

        vdata = tdata[1].split(delim)

        if vdata[0] not in mydict[tdata[0]]:
            mydict[tdata[0]][vdata[0]] = dict()

        if len(vdata)==1:
            vdata.append('total')
        mydict[tdata[0]][vdata[0]][vdata[1]] = i[1]

    return mydict

def handle0FreqProb(dict_orig):
    
    dict0 = dict_orig.copy()
    targetLevels = len(count_c)
    
    counter = 0    
    for item in dict0:
        for val in dict0[item]:
            for tval in count_c:
                if tval not in dict0[item][val]:
                    dict0[item][val][tval] = 0 
                    counter += 1

    for item in dict0:
        for val in dict0[item]:
            for tval in dict0[item][val]:
                if tval == 'total':
                    dict0[item][val][tval] += (counter*targetLevels)
                else:
                    dict0[item][val][tval] += counter
    
       
    return dict0

def predict(row):
    tokens = row.split(',')
    
    probs = dict()
    for val in prob_c:
        probs[val] = 1
    
    for col in catCols:
        if col == pred_var:
            for tval in prob_c:
                probs[tval] *= prob_c[tval]
        else:            
            for tval in prob_c:
                probs[tval] *= pdict[col][tokens[catCols[col]]][tval]
    
    #Normalization
    denominator = 0
    for item in probs:
        denominator +=  probs[item]
    
    greatestClass = -1
    greatestVal = 0
    
    for item in probs:
        probs[item] /= denominator
        if probs[item] > greatestVal:
            greatestVal = probs[item]
            greatestClass = item
    
    #return item
    return greatestClass
    
if __name__ == "__main__":
    #trainFile = 'D:\\Study\\1 DSBA\\Sem II\\Cloud Computing\\project\\work\\data\\trdata.csv'
    trainFile = sys.argv[1]
    #spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()
    
    sc = SparkContext(appName="NB")
    #spark = SparkSession.builder.appName("NaiveBayes").getOrCreate()

    #Collect frequency of each occurance of categorical values including target and store in tuples
    tuples = sc.textFile(trainFile).flatMap(lambda x: processRow(x)).reduceByKey(lambda val1,val2: val1+val2)
    
    sdict = sortAndGrpTuples(tuples.collect())
    #print(sdict)
    
    #Extract only target value details
    #Loop to find out total rows in data and target values counts
    for item in sdict[pred_var]:
        totalRows += sdict[pred_var][item][item]
        count_c[item] = sdict[pred_var][item][item]
    
    for item in count_c:
        prob_c[item] = float(count_c[item])/float(totalRows)
    
    cleanedSdict = handle0FreqProb(sdict)
    for item in sdict[pred_var]:
        totalRows += sdict[pred_var][item][item]
        count_c[item] = sdict[pred_var][item][item]
    
    for item in count_c:
        prob_c[item] = float(count_c[item])/float(totalRows)    
      
    #print(count_c)
    #print(prob_c)
    
    pdict = cleanedSdict.copy()
    cond_probs = dict()
    #print(pdict)
    

    #Calculate conditional probabilities.
    for item in pdict:
        for val in pdict[item]:
            for entry in pdict[item][val]:
                if entry not in 'total':
                    pdict[item][val][entry] /= float(count_c[entry])
    #print(pdict)
    
    print('Training Complete!!')
    #Prediction Phase begin
    #testFile = 'D:\\Study\\1 DSBA\\Sem II\\Cloud Computing\\project\\work\\data\\testsamp.csv'
    #testFile = 'D:\\Study\\1 DSBA\\Sem II\\Cloud Computing\\project\\work\\data\\tsdata.csv'
    testFile = sys.argv[1]
    predictions = sc.textFile(testFile).map(lambda x: predict(x))
    output = predictions.collect()
    import pandas as pd
    tst = pd.read_csv(testFile,header=None)

    print ('Naive Bayes Model Accuracy: ',end='')
    print((tst[5] == output).sum()/tst.shape[0])
    sc.stop()