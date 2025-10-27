from stab_generator import *
import numpy as np
from scipy.optimize import nnls
from itertools import product
import json
import time
from numpy.random import default_rng
_RAND = default_rng()

class RebitSystem:
    def __init__(self, n):
        self.n = n
        self.dim = 2**n
        
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.pauli = {'I': self.I, 'X': self.X, 'Y': self.Y, 'Z': self.Z}
        self.real_terms = {}
        
        for pauli_string in product('IXYZ', repeat=n):
            label = ''.join(pauli_string)
            matrices = [self.pauli[p] for p in pauli_string]
            term = matrices[0]
            for mat in matrices[1:]:
                term = np.kron(term, mat)
            if np.all(term.imag <= 1e-8):
                self.real_terms[label] = term
        first_key = next(iter(self.real_terms))
        self.real_terms.pop(first_key)
        self.stabilizer_basis = stabilizer_state(n)
        self.basis = []
        for alpha in self.stabilizer_basis:
            self.basis.append(self.state_to_density(alpha))

    def state_to_density(self, alpha):
        alpha = np.array(alpha, dtype=complex)
        alpha /= np.linalg.norm(alpha)
        return np.outer(alpha, alpha.conj())
    
    def is_stabilizer(self, rho, tol=1e-8):
        target_vec = rho.flatten()
        basis_vectors = [m.flatten() for m in self.basis]
        
        A = np.column_stack(basis_vectors)
        A_real = np.vstack([np.real(A), np.imag(A)])
        b_real = np.hstack([np.real(target_vec), np.imag(target_vec)])

        coefficients, residual = nnls(A_real, b_real)
        reconstruction_error = np.linalg.norm(A @ coefficients - target_vec)
        if reconstruction_error > tol:
            return False
        return True

    def random_state(self):
        X = np.sum(_RAND.normal(size=((self.dim,self.dim) + (2,))) * np.array([1, 0j]), axis=-1)
        rho = np.dot(X, X.T.conj())
        rho /= np.trace(rho)
        return rho
        
    def build_projectors(self, n_qubits):
        ket_p = np.array([1, 1]) / np.sqrt(2)
        ket_n = np.array([1, -1]) / np.sqrt(2)
        ket_0 = np.array([1, 0])
        ket_1 = np.array([0, 1])

        hadamard_basis = {'+': ket_p, '-': ket_n}
        computational_basis = {'0': ket_0, '1': ket_1}

        hadamard_projectors = []
        computational_projectors = []

        for bits in product('+-', repeat=n_qubits):
            ket = hadamard_basis[bits[0]]
            for b in bits[1:]:
                ket = np.kron(ket, hadamard_basis[b])
            proj = np.outer(ket, ket.conj())
            hadamard_projectors.append(proj)

        for bits in product('01', repeat=n_qubits):
            ket = computational_basis[bits[0]]
            for b in bits[1:]:
                ket = np.kron(ket, computational_basis[b])
            proj = np.outer(ket, ket.conj())
            computational_projectors.append(proj)

        return hadamard_projectors, computational_projectors

    def generate_q_matrix(self, rho, n_qubits):
        projectors, basis = self.build_projectors(n_qubits)
        size = len(projectors)
        Q = np.zeros((size, size), dtype=complex)
        
        for i, P in enumerate(projectors):
            for j, B in enumerate(basis):
                Q[i,j] = np.trace(P @ B @ rho)
                
        return Q

    def is_positive(self, Q, tol=1e-3):
        return np.all(np.real(Q) >= 0) and np.all(np.abs(np.imag(Q)) <= tol)

system = RebitSystem(2)

def run_simulation(n_rebits, n_samples=6600000000, progress_interval=10000):
    results = {
        'stab-pos': 0,
        'stab-neg': 0,
        'magic-pos': 0,
        'magic-neg': 0
    }

    start_time = time.time()

    for i in range(1, n_samples + 1):
        rho = system.random_state()
        q_matrix = system.generate_q_matrix(rho, n_rebits)
        is_pos = system.is_positive(q_matrix)

        if system.is_stabilizer(rho):
            key = 'stab-pos' if is_pos else 'stab-neg'
        else:
            key = 'magic-pos' if is_pos else 'magic-neg'

        results[key] += 1

        if i % progress_interval == 0 or i == n_samples:
            elapsed = time.time() - start_time
            rate = elapsed / i
            remaining = (n_samples - i) * rate

            print(f"[{i}/{n_samples}] states processed. "
                  f"Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s", flush=True)
            print("Current counts:", results, flush=True)

            temp_stats = {}
            for k in results:
                p = results[k] / i
                temp_stats[k] = {
                    'mean': p,
                    'bernoulli_std': bernoulli_std(p, i),
                    'count': results[k]
                }

            with open('gini-data2_intermediate.json', 'w') as fp:
                json.dump(temp_stats, fp, indent=4)

    final_stats = {}
    for k in results:
        p = results[k] / n_samples
        final_stats[k] = {
            'mean': p,
            'bernoulli_std': bernoulli_std(p, n_samples),
            'count': results[k]
        }

    print(f"\nFinal results for {n_rebits}-rebit system:", flush=True)
    for k, v in final_stats.items():
        print(f"{k}: {v['mean']*100:.2f}% ± {v['bernoulli_std']*100:.2f}%", flush=True)

    with open('gini-data109.json', 'w') as fp:
        json.dump(final_stats, fp, indent=4)

    return final_stats

def bernoulli_std(p, n):
    return (p * (1 - p) / n) ** 0.5

run_simulation(2)