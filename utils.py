from dscribe.descriptors import ACSF, MBTR, LMBTR, SOAP
import numpy as np
def get_embedding(obj, type=None):
    species = [3, 6, 7, 8]
    if type == 'soap_mean':
        soap = SOAP(
            species=species,
            periodic=False,
            rcut=6,
            nmax=8,
            lmax=6,

        )
        system_embedding = soap.create(obj)
        return np.mean(system_embedding, axis=0)

    elif type == 'soap_concat':

        soap = SOAP(
            species=species,
            periodic=False,
            rcut=6,
            nmax=8,
            lmax=6,

        )
        system_embedding = soap.create(obj)
        return system_embedding.reshape(system_embedding.shape[0] * system_embedding.shape[1])

    elif type == 'acsf':
        descriptor = ACSF(
            periodic=False,
            species=species,
            rcut=6,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
        )
        system_embedding = descriptor.create(obj)
        return np.mean(system_embedding, axis=0)

    elif type == 'mbtr':
        descriptor = MBTR(species=species,
                          k1={
                              "geometry": {"function": "atomic_number"},
                              "grid": {"min": 0, "max": 8, "n": 100, "sigma": 0.1},
                          },
                          k2={
                              "geometry": {"function": "inverse_distance"},
                              "grid": {"min": 0, "max": 1, "n": 100, "sigma": 0.1},
                              "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
                          },
                          k3={
                              "geometry": {"function": "cosine"},
                              "grid": {"min": -1, "max": 1, "n": 100, "sigma": 0.1},
                              "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
                          },
                          periodic=False,
                          normalization="l2_each", )
        system_embedding = descriptor.create(obj)
        return system_embedding
    if type is not None:
        raise NotImplemented
