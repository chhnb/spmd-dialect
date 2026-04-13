"""Generate libc srand(seed)+rand() sequences matching CUDA host init.
Used by C6 N-body and C7 SPH to produce identical particle positions
as the CUDA benchmarks."""
import numpy as np
import subprocess, tempfile, os

def crand_sequence(seed, n):
    """Generate n float values from libc srand(seed)+rand()/RAND_MAX."""
    src = f'#include <stdio.h>\n#include <stdlib.h>\nint main(){{srand({seed});for(int i=0;i<{n};i++)printf("%.10f\\n",(float)rand()/RAND_MAX);return 0;}}'
    with tempfile.NamedTemporaryFile(suffix='.c', delete=False, mode='w') as f:
        f.write(src)
        src_path = f.name
    bin_path = src_path.replace('.c', '')
    try:
        subprocess.run(['gcc', '-o', bin_path, src_path], check=True, capture_output=True)
        r = subprocess.run([bin_path], capture_output=True, text=True, check=True)
        return np.array([float(x) for x in r.stdout.strip().split('\n')], dtype=np.float32)
    finally:
        for p in [src_path, bin_path]:
            if os.path.exists(p):
                os.unlink(p)


def nbody_init(N, seed=42):
    """Generate N-body init matching CUDA nbody_benchmark.cu srand(42):
    positions in [-1,1]^3, zero velocity."""
    vals = crand_sequence(seed, N * 3)
    pos = (vals.reshape(N, 3) * 2 - 1)
    return pos


def sph_init(N, seed=42, domain=1.0):
    """Generate SPH init matching CUDA sph_benchmark.cu srand(42):
    positions in [0, domain]^2."""
    vals = crand_sequence(seed, N * 2)
    pos = vals.reshape(N, 2) * domain
    return pos
