import numpy as np
import cdd
from twoqubitrealstabilizers import Pauli_rep_real, Pauli_rep_css, Pauli_rep_stab, the_order
from scipy.optimize import linprog
import csv

VERTICES_REBIT = Pauli_rep_real
VERTICES_CSS = Pauli_rep_css
VERTICES_STAB = Pauli_rep_stab

def vertices_to_faces(vertices):
    """
    Uses cdd to calcualte the convex hull of the given vertices and
    returns the polytope in terms of an H-space represenation.

    Note, to get the array use the output.array attribute. And use
    output.lin_set to find the set of equations that are equalities
    (instead of inequalities).
    """

    #Pad the first column of the vertices with 1s to indicate they are vetices, not rays
    A = np.hstack((np.ones((vertices.shape[0], 1)), vertices))

    # use double description to find H rep
    P = cdd.polyhedron_from_matrix(cdd.matrix_from_array(A, rep_type = cdd.RepType.GENERATOR))

    # Extract the H-space data
    H = cdd.copy_inequalities(P)

    return H


def check_h_rep(vertices, H_rep, tol = 1e-9):
    """
    Checks if each vertex indeed stasfies all the inequalties.
    """

    violation_found = False

    inequalities = H_rep.array

    for vertex in vertices:
        for j, ineqaulity in enumerate(inequalities):
            RHS = np.dot(ineqaulity, vertex)
            if j in H_rep.lin_set:
                if not np.abs(RHS) < tol:
                    violation_found = True
            else:
                if not RHS > -tol:
                    violation_found = True
    
    return violation_found


def Pauli_2_Density(r):
    """
    Convert Pauli rep to density matrix rep
    """
    rho = np.zeros((4,4), dtype= complex)
    for j, coordinate in enumerate(r):
        rho += coordinate*the_order[j]
    return rho

def calculate_KD_distribution(rho, U):
    Q = np.zeros((4,4), dtype = complex)
    for j in range(4):
        for k in range(4):
            for l in range(4):
                Q[j][k] += U[j][k]*U[l][k]*rho[j][l]
    return Q

def check_if_KD_positive(rho,U, tol = 1e-9):
    Q = calculate_KD_distribution(rho, U)
    return np.all(Q >= -tol)

def check_if_magic(rho):
    """
    Checks if a 2-qubit state is magic
    """
    STAB = []
    for r in VERTICES_STAB:
        STAB.append(Pauli_2_Density(r).flatten())

    Ar = np.real(np.transpose(STAB)).tolist()
    Ai = np.imag(np.transpose(STAB)).tolist()
    A = Ar + Ai + [[1]*len(STAB)]
    b = np.real(rho.flatten()).tolist() + np.imag(rho.flatten()).tolist() + [1]  

    n_vars = np.array(A).shape[1]

    result = linprog(np.zeros(n_vars), A_ub= None, b_ub=None, A_eq=np.array(A), b_eq=np.array(b), bounds=[(0, None)] * n_vars, method='highs') 

    return not result.success

def remove_indices(data, indices_to_remove):
    """
    Removes elements from `data` at the positions specified in `indices_to_remove`.

    Parameters:
    - data: list of lists
    - indices_to_remove: set of integers (indices to remove)

    Returns:
    - A new list with the specified indices removed.
    """
    return [item for i, item in enumerate(data) if i not in indices_to_remove]

def write_list_of_lists_to_csv(data, filename):
    """
    Writes a list of lists to a CSV file.

    Parameters:
    - data: list of lists (rows to write)
    - filename: string (name of the output CSV file)
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)




def generate_faces():
    # We start by finding all the faces of the CSS, REBIT, and STAB polytopes
    # And check how many faces we've found
    h_rep_CSS = vertices_to_faces(VERTICES_CSS)
    print('Got h rep for CSS')
    print(f'We found {len(h_rep_CSS.array)} faces')
    print(f'Of which {len(h_rep_CSS.lin_set)} are equalities')
    CSS_faces = remove_indices(h_rep_CSS.array, h_rep_CSS.lin_set)
    write_list_of_lists_to_csv(CSS_faces, 'CSS_faces.csv')


    h_rep_REBIT = vertices_to_faces(VERTICES_REBIT)
    print('Got h rep for REBIT')
    print(f'We found {len(h_rep_REBIT.array)} faces')
    print(f'Of which {len(h_rep_REBIT.lin_set)} are equalities')
    print(f'the equality is constraint {h_rep_REBIT.lin_set}')
    REBIT_faces = remove_indices(h_rep_REBIT.array, h_rep_REBIT.lin_set)
    write_list_of_lists_to_csv(REBIT_faces, 'REBIT_faces.csv')

    h_rep_STAB = vertices_to_faces(VERTICES_STAB)
    print('Got h rep for STAB')
    print(f'We found {len(h_rep_STAB.array)} faces')
    print(f'Of which {len(h_rep_STAB.lin_set)} are equalities')
    print(f'the equality is constraint {h_rep_STAB.lin_set}')
    STAB_faces = remove_indices(h_rep_STAB.array, h_rep_STAB.lin_set)
    write_list_of_lists_to_csv(STAB_faces, 'STAB_faces.csv')



def catagorize_faces():
    CSS_faces =  np.loadtxt('CSS_faces.csv', delimiter=',', dtype=float)
    REBIT_faces =  np.loadtxt('REBIT_faces.csv', delimiter=',', dtype=float)
    STAB_faces =  np.loadtxt('STAB_faces.csv', delimiter=',', dtype=float)

    CSS_REBIT_STAB_faces = []
    tol = 1e-9


    for face in CSS_faces:
        if np.any(np.linalg.norm(REBIT_faces - face, axis=1) < tol):# and np.any(np.linalg.norm(STAB_faces - face, axis=1) < tol):
            CSS_REBIT_STAB_faces.append(face)
    
    # print(len(CSS_REBIT_STAB_faces))
    return CSS_REBIT_STAB_faces

def find_switch_point(f, x_max=1.0, tol=1e-16, max_iter=100000):
    """
    Find the smallest x > 0 such that f(x) != f(0), using bisection.

    Parameters:
    - f: callable, boolean function f(x)
    - x_max: upper bound to search in (must contain the switch)
    - tol: desired precision
    - max_iter: max bisection iterations

    Returns:
    - x such that f(x - eps) == f(0), f(x) != f(0), and x <= x_max within tolerance
    """

    f0 = f(0)
    # Step 1: Ensure switch exists in (0, x_max]
    if f(x_max) == f0:
        raise ValueError("No switch detected within x_max. Try increasing x_max.")

    # Step 2: Bisection
    left = 0.0
    right = x_max

    for _ in range(max_iter):
        mid = (left + right) / 2
        if f(mid) == f0:
            left = mid
        else:
            right = mid
        if right - left < tol:
            return right

    raise RuntimeError("Switch point not found within max_iter iterations.")

def check_if_semipositive_definite(rho):
    evs,_ = np.linalg.eig(rho)
    return np.all(evs>=0)




if __name__ == '__main__':

    U = np.array([[ 1, 1, 1, 1],
                  [ 1,-1, 1,-1],
                  [ 1, 1,-1,-1],
                  [ 1,-1,-1, 1]])/2
    
    F = np.array([[ 1, 0, 1, 1],
                  [ 0, 1,-1,-1],
                  [ 1,-1,-1,-2],
                  [ 1,-1,-2,-1]])

    CSS_REBIT_faces = catagorize_faces()
    # print(len(CSS_REBIT_faces))
    CSS_faces =  np.loadtxt('CSS_faces.csv', delimiter=',', dtype=float)
    REBIT_faces =  np.loadtxt('REBIT_faces.csv', delimiter=',', dtype=float)
    bound_States_counter = 0

    for j in range(len(CSS_REBIT_faces)):
        print('\n')
        print(f'Now testing shared face {j}')
        def f(l):
            constant = CSS_REBIT_faces[j][0]
            normal = CSS_REBIT_faces[j][1:]

            # normal = np.array([0.0,0.0,0.0,0.0,4.0,-4.0,0.0,-4.0,-4.0,-4.0,0.0,0.0,0.0,0.0,0.0,0.0])
            normal = -l*normal
            normal[0] = 1/4

            rho = Pauli_2_Density(normal)
            # rho = np.eye(4)/4 + l*F

            return check_if_KD_positive(rho, U)

        def g(l):
            constant = CSS_REBIT_faces[j][0]
            normal = CSS_REBIT_faces[j][1:]

            # normal = np.array([0.0,0.0,0.0,0.0,4.0,-4.0,0.0,-4.0,-4.0,-4.0,0.0,0.0,0.0,0.0,0.0,0.0])
            normal = -l*normal
            normal[0] = 1/4

            rho = Pauli_2_Density(normal)
            # rho = np.eye(4)/4 + l*F
            return check_if_magic(rho)

        def h(l):
            constant = CSS_REBIT_faces[j][0]
            normal = CSS_REBIT_faces[j][1:]

            # normal = np.array([0.0,0.0,0.0,0.0,4.0,-4.0,0.0,-4.0,-4.0,-4.0,0.0,0.0,0.0,0.0,0.0,0.0])
            normal = -l*normal
            normal[0] = 1/4

            rho = Pauli_2_Density(normal)
            # rho = np.eye(4)/4 + l*F
            return check_if_semipositive_definite(rho)

        KDp_switch = np.round(find_switch_point(f, tol = 1e-15),10)
        magic_switch = np.round(find_switch_point(g, tol = 1e-15),10)
        SD_switch = np.round(find_switch_point(h, tol = 1e-15),10)
        print(f'The state stops being KD positive at {KDp_switch}.')
        print(f'The state stops being positive semidefnite at {SD_switch}.')
        print(f'The state starts being magic at {magic_switch}.')
        if magic_switch < KDp_switch:
            print('We have thus found a bound state')
            bound_States_counter += 1

    print(f'We have found a total of {bound_States_counter} bound states.')

