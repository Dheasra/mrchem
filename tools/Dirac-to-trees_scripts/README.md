# Scripts to translate H5 binaries representing basis set and coefficent matrices outputted by DIRAC.

Currently limited is scope, expects a "H.h5" file existing in the same directory as the scripts.
Can only translate s orbtials with confidence for now.

# How to use 
both scripts require the h5py package and scipy.

## h5_to_coeff.py
h5_to_coeff is called as

```bash
$ python h5_to_coeff.py filename.h5 n_pairs
```

Currently the input h5 binary is only fetched in the same directory as the script. By default it tries to fetch H.h5.
n_pairs represents the number of spinor pairs you want to extract, starting from the lowest energy ones (bound states only).
By default, n_pairs is set to fetch all spinors. 
The script will fail if you try to fetch only a part of a filled shell, for example n_pairs = 4 will fail,
because you would be attempting to fetch the 1s 2s 2p1/2 pairs and half of the 2p3/2 shell (which contains 4 electrons).

## h5_to_bas.py

h5_to_bas is called as

```bash
$ python h5_to_bas.py
```

Currently it has no input variable to indicate the filename, you need to change the name directly in the script.