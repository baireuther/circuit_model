# Readme for circuit_model.py

The code in circuit_model.py implements a simplified Pauli error channel model for odd distance, square shaped surface code with rough edges [1]. The error model follows the circuit model described in [2] with some deviations which are discussed in [3].
In particular it does not include correlated two-qubit errors during the CNOT gates.
To avoid hook errors it uses the improvements suggested in [4].
The code layout and circuit are for example illustrated in figure 1 of [3].

The version that is discussed in Ref. [3] is on branch arXiv1705p07855.

## How to use the code to generate the data used in Figure 3 of Ref. [3]
1) In circuit_model.py, set the path (db_path) where the databases with the
   results shall be written to.  
   WARNING: existing databases will be overwritten!
2) Set the variable mode in circuit_model.py to  
   0 if you want the training data, or  
   1 if you want the validation data, or  
   2 if you want the test data.
3) python circuit_model.py


## How to use the code to generate the data used in Figure 4 of Ref. [3]
1) In circuit_model.py, set the path (db_path) where the databases with the
   results shall be written to.   
   WARNING: existing databases will be overwritten!
2) Set the variable fy in circuit_model.py to 0, 0.5, 1.0, 1.5, or 2.0
   corresponding to the different y-error rates from left to right in Figure 4
   of Ref. [3].  
3) Set the variable mode in circuit_model.py to  
   0 if you want the training data, or  
   1 if you want the validation data, or  
   2 if you want the test data.
4) python circuit_model.py


## References
[1] H. Bombin and M. A. Martin-Delgado, Phys. Rev. A 76, 012305 (2007)  
[2] A. G. Fowler, M. Mariantoni, J. M. Martinis, and A. N. Cleland, Phys. Rev. A 86, 032324 (2012)  
[3] P. Baireuther, T. E. O'Brien, B. Tarasinski, and C. W. J. Beenakker, Quantum 2, 48 (2018)  
[4] Y. Tomita and K. M. Svore, Phys. Rev. A 90, 062320 (2014)
