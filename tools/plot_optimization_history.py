import pickle

import matplotlib.pyplot as plt
import tqdm
from frmax2.core import DistributionGuidedSampler
from frmax2.region import SuperlevelSet

from pr2dmp.demonstration import project_root_path

project_name = "fridge_door_open"
sampler_cache_dir = project_root_path(project_name) / "sampler_cache"

opt_params = []
indices = range(0, 290, 10)
for i in tqdm.tqdm(indices):
    with open(sampler_cache_dir / f"cache_{i}.pkl", "rb") as f:
        cache: DistributionGuidedSampler = pickle.load(f)

    opt_param_cache_path = sampler_cache_dir / f"opt_param_{i}.pkl"
    if opt_param_cache_path.exists():
        with open(opt_param_cache_path, "rb") as f:
            opt_param = pickle.load(f)
    else:
        opt_param = cache.optimize(2000, method="cmaes")
        with open(opt_param_cache_path, "wb") as f:
            pickle.dump(opt_param, f)
        print(f"saving {opt_param_cache_path}")
    opt_params.append(opt_param)

assert isinstance(cache.fslset, SuperlevelSet)
param_indices = list(range(len(opt_param + 1)))
values = []
for opt_param in opt_params:
    value = cache.fslset.sliced_volume_mc(opt_param, param_indices, 1000)
    values.append(value)

plt.plot(values)
plt.show()
