This folder collects code to replicate the experiments discussed in the paper "Bayesian Nonparametrics Meets Data-Driven Distributionally Robust Optimization" by Nicola Bariletto and Nhat Ho.


The script "BNP_DRO_functions.py" contains the main functions needed to implement the experiments (e.g., the ones to construct the DP based criterion, the SGD algorithm, etc.). The script also contains code to import the utilities necessary to run the other scritps (see below).


Each one of the other scripts implements one of the experiments presented in the paper (the titles of the script should make it clear which experiment it refers to). At the beginning of each script, the line

exec(open("yourpath/BNP_DRO_functions.py").read())

appears. Please replace "yourpath" with your local path at which you saved the script "BNP_DRO_functions.py". After this replacement, as long as every utility imported in the script "BNP_DRO_functions.py" is installed, the code in each experiment script is self-contained and runs from beginning to end without the need of any other modification. The choice of random seed also ensures full replicability of our results.
