import numpy as np
import joblib
from nilearn.connectome import ConnectivityMeasure
from pathlib import Path
import argparse


def generate_fc(path, kind='correlation', vectorize=True, discard_diagonal=True):
    """
    Generate functional connectivity matrix from 4D fMRI data
    :param arr: 4D fMRI data
    :param kind: type of functional connectivity matrix
    :return: functional connectivity matrix
    """
    arr = np.loadtxt(path)
    conn = ConnectivityMeasure(
        kind=kind, vectorize=vectorize, discard_diagonal=discard_diagonal)
    fc = conn.fit_transform([arr])[0]
    return fc


def main(args):
    path = Path(args.path)
    output = Path(args.output)
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    files = path.glob('*.1D')

    for file in files:
        fc = generate_fc(file)
        np.savetxt(output.joinpath(file.name), fc, fmt='%.4f')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate functional connectivity matrix from atlas fMRI data')
    parser.add_argument('--path', type=str, help='path to fMRI data')
    parser.add_argument(
        '--output', type=str, help='path to output functional connectivity matrix directory')

    args = parser.parse_args()
    main(args)
