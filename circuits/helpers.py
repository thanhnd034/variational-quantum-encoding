import numpy as np

def normalize_amplitude(v):
    arr = np.array(v, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Cannot amplitude-normalize: vector norm is zero.")
    return (arr / norm).tolist()

def get_a(ip):
    a = []
    for i in range(len(ip)):
        a.append((ip[i])**2)
    return a

def normalize_sum(v):
    # Convert to float to avoid integer division issues
    v = [float(x) for x in v]
    
    s = sum(v)
    if s == 0:
        raise ValueError("Cannot normalize: sum of vector components is zero.")
    
    return [x / s for x in v]

def get_Ry(x0,x1):
    r = np.sqrt((x0**2)+(x1**2))
    if r > 0:
        alpha = 2*np.arcsin(x1/r)
    else:
        alpha = 2*np.pi - 2*np.arcsin(x1/r)
    return r, alpha

def calculateRy(a):
    angles = []
    cal = a
    while len(cal) != 1:
        cal1 = cal
        cal = []
        for i in range(int(len(cal1)/2)):
            theta = 0
            r, theta = get_Ry(cal1[2*i], cal1[2*i+1])
            cal.append(r)
            angles.append(theta)
    return angles

def repositionVector(a):
    b = []
    N = len(a) + 1
    i = 1
    while i != N:
        b.append(a[(N-2*i):(N-i)])
        i = i*2
    flat = np.concatenate(b)
    return flat
    
def int_to_reversed_bitstring(n, num_bits=None):
    if num_bits is None:
        bitstring = format(n, "b")        # no padding
    else:
        bitstring = format(n, f"0{num_bits}b")
    return bitstring[::-1]

def make_list(i):
    return list(range(i+1))

def processData(ip):
    b = normalize_amplitude(ip)
    a = get_a(b)
    return a,b

def generate_random_data(N, seed, low=-1.0, high=1.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, 2**N)

