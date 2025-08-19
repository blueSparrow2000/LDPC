# LDPC recovery
This project aims to recover H matrix of a LDPC code without candidate set on a noisy channel      

Final goal is to fully recover Parity Check Matrix(H) of a NAND flash memory      

[Theoretical background][link]

[link]:https://bluesparrow2000.github.io/paperreview/LDPC/


## Installations
This project is based on python. Below are the packages that needs to be installed:

numpy      
numba      
scipy              

## Files
- main.py      
An executable file that generates LDPC code with random H matrix, and performs ECO to recover H matrix 
- progressive_reconstruction.py
An executable file based on paper 'progressive reconstruction of LDPC H matrix ...'
- column_extract_reconstruction.py
An executable file based on paper about 'column extraction method' by Y.ding
- GERCE.py
An implementation of function in column extraction method
- LDPC_sampler.py      
Randomly generates H matrix and code words corresponding to it 
- ECO.py      
Performs fast Elementary Column Operation
- gauss_elim.py      
Original code for fast gaussian elimination on GF2 (binary matrix)
- extracter.py       
Extracts parity check vector from ECO matrix                  
- verifier.py               
Format H matrix into diagonal format and verify if it is same as the original one     
- sparsifier.py        
Sparsify a binary matrix       
- prettyprinter.py       
Prints 2D array in a readable format     
- fullcode.py           
One file that contains everything, used in translation to C code

## version history
2025.06.24 Basic LDPC simulation without noise        
2025.07.03 Add small noise and sort dual vectors by reliability (if more dual vectors are found than n-k, we crop it)
2025.07.22 ~  2026.07.25 Progressive reconstruction implementation
2025.07.31 Column extraction method implementation 
2025.08.13 ECO parallelism to full capacity
2025.08.17 Debug ECO for GERCE file
2025.08.19 CSV file saver

TODO
Read the real codeword data and try to recover H matrix
Find optimal beta value


## License
Available for non-commercial use      