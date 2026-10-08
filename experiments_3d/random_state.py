"""Validation must not change the subsequent training random stream."""
from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def isolated_evaluation(learner,seed):
    python_state=random.getstate()
    numpy_state=np.random.get_state()
    torch_state=torch.get_rng_state()
    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rng=getattr(learner,'rng',getattr(learner,'_rng',None))
    local_state=rng.getstate() if rng is not None else None
    generator=getattr(learner,'generator',None)
    generator_state=generator.get_state() if generator is not None else None
    try:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if rng is not None:rng.seed(seed)
        if generator is not None:generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state);torch.set_rng_state(torch_state)
        if cuda_state is not None:torch.cuda.set_rng_state_all(cuda_state)
        if rng is not None:rng.setstate(local_state)
        if generator is not None:generator.set_state(generator_state)
