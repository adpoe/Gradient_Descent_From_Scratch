# Linear Regression with Gradient Descent
* @author Anthony (Tony) Poerio
* @email tony@tonypoer.io

## Overview
This project contains an implementation of two Gradient Descent algorithms:  
1. Univariate Linear Regression Gradient Descent  
2. Multivariate Linear Regression Gradient Descent  

Both algorithms can be used/tested simply by passing in the correct command line arguments to the **lin_regr.py** python file.

### Source Code
The source code for my AI contained in the file named '**lin_regr.py**'

### Running the program - Univariate Regression
To run the program for Univariate Regression, you must pass in **TWO command line arguments**.

- The first argument must a CSV file containing the TRAINING DATA  
- The second Argument must be a CSV file containing the TEST DATA  

For example you can run the univariate regression gradient descent algorithm by using this command:  
`python lin_regr.py "part1 test.csv" "part2 test.csv"`

TO NOTE: --> You only need to use quotes around the csv file names if they have SPACE in them. The example files did have a space, so I am including in the specification here, so there is no confusion in getting up and runnning. 

### Running the program - Multivariate Regression
To run the program for Multivariate Regression, you must pass in **ONE command line argument**.  

- The ONLY argument must a CSV file containing THE FULL DATA SET  

This data set will be RANDOMLY DIVIDED into a **training set (80% of the data)**, and a **test set (20% of the data)**.  

Occasionally, this division into training/test set will FAIL, because I am using an ASSERTION to make sure we have division reasonably close to 80/20. If this happens **PLEASE RE-RUN THE PROGRAM**. This check is only to ensure that we have enough randomly selected testing data to achieve accurate test results.

For example you can run the multivariate regression gradient descent algorithm by using this command:  
`python lin_regr.py part2.csv`


### Results
Whichever version of the program is run, the results will be printed via stdout.  

### Version
This source code is written using python version 2.7.8

--

## Prerequisities
This project depends upon Python v. 2.7.8

I am also using the **numpy** and **itertools** python libraries


## Built With
* Python v. 2.7.8
* Numpy
* Itertools
* AST
* PyCharm

