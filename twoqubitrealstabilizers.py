import numpy as np

X = np.array([[0,1],
              [1,0]])

Y = np.array([[0,-1j],
              [1j,0]])

Z = np.array([[1, 0],
              [0,-1]])

I = np.array([[1,0],
              [0,1]])

#Single qubit
S1 = [np.kron(I,I), np.kron(Z,I), np.kron(I,Z), np.kron(Z,Z)]
S2 = [np.kron(I,I), np.kron(Z,I), np.kron(I,X), np.kron(Z,X)]
S3 = [np.kron(I,I), np.kron(X,I), np.kron(I,Z), np.kron(X,Z)]
S4 = [np.kron(I,I), np.kron(X,I), np.kron(I,X), np.kron(X,X)]

#Bell
S5 = [np.kron(I,I), np.kron(X,X), np.kron(Z,Z), np.kron(-Y,Y)]

#Non CSS
S6 = [np.kron(I,I), np.kron(X,Z), np.kron(Z,X), np.kron(Y,Y)]

CSS_STABS_GENS = [S1,S2,S3,S4,S5]
REAL_STABS_GENS = [S1,S2,S3,S4,S5,S6]

the_pattern = [[1, 1, 1, 1],
               [1,-1, 1,-1],
               [1, 1,-1,-1],
               [1,-1,-1, 1]]



#rho rep
rho_rep_real = np.array([np.kron(I,I) + np.kron(I,X) + np.kron(X,I) + np.kron(X,X),
                         np.kron(I,I) - np.kron(I,X) + np.kron(X,I) - np.kron(X,X),
                         np.kron(I,I) + np.kron(I,X) - np.kron(X,I) - np.kron(X,X),
                         np.kron(I,I) - np.kron(I,X) - np.kron(X,I) + np.kron(X,X),
                         
                         np.kron(I,I) + np.kron(I,X) + np.kron(Z,I) + np.kron(Z,X),
                         np.kron(I,I) - np.kron(I,X) + np.kron(Z,I) - np.kron(Z,X),
                         np.kron(I,I) + np.kron(I,X) - np.kron(Z,I) - np.kron(Z,X),
                         np.kron(I,I) - np.kron(I,X) - np.kron(Z,I) + np.kron(Z,X),

                         np.kron(I,I) + np.kron(I,Z) + np.kron(X,I) + np.kron(X,Z),
                         np.kron(I,I) - np.kron(I,Z) + np.kron(X,I) - np.kron(X,Z),
                         np.kron(I,I) + np.kron(I,Z) - np.kron(X,I) - np.kron(X,Z),
                         np.kron(I,I) - np.kron(I,Z) - np.kron(X,I) + np.kron(X,Z),

                         np.kron(I,I) + np.kron(I,Z) + np.kron(Z,I) + np.kron(Z,Z),
                         np.kron(I,I) - np.kron(I,Z) + np.kron(Z,I) - np.kron(Z,Z),
                         np.kron(I,I) + np.kron(I,Z) - np.kron(Z,I) - np.kron(Z,Z),
                         np.kron(I,I) - np.kron(I,Z) - np.kron(Z,I) + np.kron(Z,Z),

                         np.kron(I,I) + np.kron(X,X) + np.kron(Z,Z) - np.kron(Y,Y),
                         np.kron(I,I) - np.kron(X,X) + np.kron(Z,Z) + np.kron(Y,Y),
                         np.kron(I,I) + np.kron(X,X) - np.kron(Z,Z) + np.kron(Y,Y),
                         np.kron(I,I) - np.kron(X,X) - np.kron(Z,Z) - np.kron(Y,Y),

                         np.kron(I,I) + np.kron(X,Z) + np.kron(Z,X) + np.kron(Y,Y),
                         np.kron(I,I) - np.kron(X,Z) + np.kron(Z,X) - np.kron(Y,Y),
                         np.kron(I,I) + np.kron(X,Z) - np.kron(Z,X) - np.kron(Y,Y),
                         np.kron(I,I) - np.kron(X,Z) - np.kron(Z,X) + np.kron(Y,Y),
                         ])


rho_rep_css  = np.array([np.kron(I,I) + np.kron(I,X) + np.kron(X,I) + np.kron(X,X),
                         np.kron(I,I) - np.kron(I,X) + np.kron(X,I) - np.kron(X,X),
                         np.kron(I,I) + np.kron(I,X) - np.kron(X,I) - np.kron(X,X),
                         np.kron(I,I) - np.kron(I,X) - np.kron(X,I) + np.kron(X,X),
                         
                         np.kron(I,I) + np.kron(I,X) + np.kron(Z,I) + np.kron(Z,X),
                         np.kron(I,I) - np.kron(I,X) + np.kron(Z,I) - np.kron(Z,X),
                         np.kron(I,I) + np.kron(I,X) - np.kron(Z,I) - np.kron(Z,X),
                         np.kron(I,I) - np.kron(I,X) - np.kron(Z,I) + np.kron(Z,X),

                         np.kron(I,I) + np.kron(I,Z) + np.kron(X,I) + np.kron(X,Z),
                         np.kron(I,I) - np.kron(I,Z) + np.kron(X,I) - np.kron(X,Z),
                         np.kron(I,I) + np.kron(I,Z) - np.kron(X,I) - np.kron(X,Z),
                         np.kron(I,I) - np.kron(I,Z) - np.kron(X,I) + np.kron(X,Z),

                         np.kron(I,I) + np.kron(I,Z) + np.kron(Z,I) + np.kron(Z,Z),
                         np.kron(I,I) - np.kron(I,Z) + np.kron(Z,I) - np.kron(Z,Z),
                         np.kron(I,I) + np.kron(I,Z) - np.kron(Z,I) - np.kron(Z,Z),
                         np.kron(I,I) - np.kron(I,Z) - np.kron(Z,I) + np.kron(Z,Z),

                         np.kron(I,I) + np.kron(X,X) + np.kron(Z,Z) - np.kron(Y,Y),
                         np.kron(I,I) - np.kron(X,X) + np.kron(Z,Z) + np.kron(Y,Y),
                         np.kron(I,I) + np.kron(X,X) - np.kron(Z,Z) + np.kron(Y,Y),
                         np.kron(I,I) - np.kron(X,X) - np.kron(Z,Z) - np.kron(Y,Y)
                         ])

#Pauli Rep
#                           II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
Pauli_rep_real = np.array([[ 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 1,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 1, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 0, 0, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 0, 0, 1, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0,-1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0,-1, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 0,-1, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 0, 1, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 0,-1, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0]
                          ])/4
                      

#                           II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
Pauli_rep_css  = np.array([[ 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 1,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 1, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 0, 0, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],

                           [ 1, 0, 0, 0, 1, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0,-1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0,-1, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0],
                          ])/4
                      
#                           II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
Pauli_rep_stab = np.array([
                           #<IX,XI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 1, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                           #<IX,ZI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 1, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1,-1, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0],

                           #<IZ,XI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 1,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

                           #<IZ,ZI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 1, 0, 0, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                           [ 1, 0,-1, 0, 0, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0, 0],

                           #<XX,ZZ>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 1, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0,-1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 1, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0,-1, 0, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0],

                           #<XZ,ZX>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 0,-1, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 0, 1, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0],
                           [ 1, 0, 0, 0, 0,-1, 0,-1, 0, 1, 0, 0, 0, 0, 0, 0],

                           #<ZI,IY>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                           [ 1, 0, 0, 0, 0, 0,-1, 0, 0, 0, 1, 0,-1, 0, 0, 0],
                           [ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1, 0,-1, 0, 0, 0],
                           [ 1, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 1, 0, 0, 0],

                           #<XI,IY>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                           [ 1, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0],
                           [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 0, 0],
                           [ 1, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0],

                           #<IZ,YI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                           [ 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1],
                           [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0,-1],
                           [ 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 1],
                        
                           #<IX,YI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                           [ 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0],
                           [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0],
                           [ 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0],

                           #<IY,YI>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                           [ 1, 0, 0, 0, 0, 0, 0, 0, 0,-1,-1, 0, 0, 1, 0, 0],
                           [ 1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0,-1, 0, 0],
                           [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,-1, 0, 0,-1, 0, 0],

                           #<XY,YX>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                           [ 1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1, 0, 0, 1, 0],
                           [ 1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1, 0, 0,-1, 0],
                           [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1, 0, 0,-1, 0],

                           #<ZY,YZ>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                           [ 1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 1],
                           [ 1, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,-1],
                           [ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0,-1],

                           #<XY,ZX>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0],
                           [ 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                           [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,-1, 0, 1, 0],
                           [ 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0,-1, 0,-1, 0],

                           #<XY,ZX>
                           #II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
                           [ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,-1],
                           [ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,-1, 0, 0, 0, 1],
                           [ 1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 1, 0, 0, 0, 1],
                           [ 1, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0,-1, 0, 0, 0,-1],
                          ])/4

#                           II,IX,IZ,XI,XX,XZ,ZI,ZX,ZZ,YY,IY,XY,ZY,YI,YX,YZ
the_order = [np.kron(I,I),
             np.kron(I,X), 
             np.kron(I,Z), 
             np.kron(X,I), 
             np.kron(X,X), 
             np.kron(X,Z), 
             np.kron(Z,I), 
             np.kron(Z,X), 
             np.kron(Z,Z), 
             np.kron(Y,Y),
             np.kron(I,Y),
             np.kron(X,Y),
             np.kron(Z,Y),
             np.kron(Y,I),
             np.kron(Y,X),
             np.kron(Y,Z)]

