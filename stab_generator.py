import numpy as np
from itertools import product

def bits2int(bits):
    r = 0
    s = 1
    for b in bits:
        if b & 1:
            r += s
        s <<= 1
    return r

def parity(v):
    return bin(v).count('1') & 1

def normalize_state(state):
    norm = np.linalg.norm(state)
    for s in state:
        if s != 0:
            norm *= s/abs(s)
            break;
    return state/norm

def state_support(state):
    n = 0
    for s in state:
        if s != 0:
            n += 1
    return n

def gf2_rank(rows):
    rank = 0
    while rows:
        pivot_row = rows.pop()
        if pivot_row:
            rank += 1
            lsb = pivot_row & -pivot_row
            for index, row in enumerate(rows):
                if row & lsb:
                    rows[index] = row ^ pivot_row
    return rank

def gf2_mul_mat_vec(m, v):
    return bits2int(map(parity, m & v))

def gf2_mul_vec_vec(v1, v2):
    return parity(v1 & v2)

def qbinomial(n, k, q = 2):
    c = 1
    for j in range(k):
        c *= q**n - q**j
    for j in range(k):
        c //= q**k - q**j
    return c

def number_of_states(n):
    return [2**n * 2**((k+1)*k//2) * qbinomial(n, k) for k in range(n+1)]

def all_possible_randint(k, n):
    return np.array(list(product(range(k), repeat=n)))

def remove_duplicate_states(states):
    unique_states = []
    seen_hashes = set()
    
    for state in states:
        normalized = state / np.linalg.norm(state)
        
        for i in range(len(normalized)):
            if abs(normalized[i]) > 1e-10:
                phase = normalized[i] / abs(normalized[i])
                normalized = normalized / phase
                break
                
        state_hash = tuple(np.round(normalized, 10))
        
        if state_hash not in seen_hashes:
            seen_hashes.add(state_hash)
            unique_states.append(state)
            
    return unique_states

def stabilizer_state(n):
    dtype = np.int32
    nmax = np.log2(np.iinfo(dtype).max + 1)
    if not (0 < n and n <= nmax):
        raise ValueError('Number of qubits must be in range(1, %d)!' % (nmax + 1))
    dimn = 2**n
    states = []
    for i in range(2**n):
        state = np.zeros(dimn, dtype=np.complex64)
        state[i] = 1
        states.append(state)
    for k in range(1,n+1):
        dimk = 2**k
        for R in [r for r in all_possible_randint(dimk, n) if gf2_rank(list(r))>=k]:
            for t in all_possible_randint(dimn, 1):
                t = int(t[0])
                for Q in all_possible_randint(dimk, k):
                    Q = Q.astype(dtype)
                    for c in all_possible_randint(dimk, 1):
                        c = int(c[0])
                        state = np.zeros(dimn, dtype=np.complex64)
                        for x in range(dimk):
                            y = gf2_mul_mat_vec(R, x) ^ t
                            ib = gf2_mul_vec_vec(c, x)
                            mb = gf2_mul_vec_vec(x, gf2_mul_mat_vec(Q, x))
                            state[y] = 1j**ib * (-1)**mb
                        state = normalize_state(state)
                        states.append(state)
    return remove_duplicate_states(states)