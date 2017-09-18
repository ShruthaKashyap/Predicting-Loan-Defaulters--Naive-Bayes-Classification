# Predicting-Loan-Defaulters--Naive-Bayes-Classification
http://webpages.uncc.edu/skashya1/


The algorithms were implemented using Python version 3.5.2 and Spark 


Running the application on cluster:

1) Put the .py files on the cluster server. 

2) Put the input files on hdfs. (Run the below scp command to to get the files onto the cluster)

   $ hadoop fs -put <cluster file paths> <hdfs path>

3) Run spark-submit on the .py file

	a. Naive Bayes
	   $ spark-submit <filename>.py <path to training data set> <path to test dataset>

	b. K Nearest Neighbor
	   $ spark-submit <filename>.py <k-value> <path to training data set> <path to test dataset>
