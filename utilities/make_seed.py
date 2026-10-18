import hashlib
from configs.settings import MASTER_SEED

def make_seed(name, master_seed=MASTER_SEED):
    msg = f"{name}_{master_seed}".encode("utf-8")
    digest = hashlib.sha256(msg).hexdigest()
    
    # Convert hex digest to integer seed within 32-bit range
    seed = int(digest, 16) % (2**32)
    return seed