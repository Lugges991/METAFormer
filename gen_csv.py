import pandas as pd
from pathlib import Path
import argparse


def main(args):
    paths = args.paths
    output = Path(args.output)

    pheno_df = pd.read_csv("args.pheno_file")
    dfs = []
    for path in paths:
        files = list(Path(path).glob('*.1D'))

        ps = []
        sids = []
        labs = []
        for file in files:
            sid = file.stem.split('_')[-3]
            sids.append(sid)
            lab = pheno_df[pheno_df['SUB_ID'] ==
                           int(sid)].DX_GROUP.values[0] - 1
            labs.append(lab)
            ps.append(file)

        atlas = ps[0].stem.split('_')[-1]
        df = pd.DataFrame({f'{atlas}': ps, 'SID': sids, 'LABELS': labs})
        dfs.append(df)
    fin_df = dfs[0]
    for df in dfs[1:]:
        fin_df = pd.merge(fin_df, df, on=['SID', 'LABELS'], how='inner')

    fin_df.to_csv(output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate csv from functional connectivity matrices of different atlases")
    # take list of paths
    parser.add_argument(
        'paths', nargs=3, help='paths to functional connectivity matrices')
    parser.add_argument('--pheno_file', type=str, help='path to pheno file')
    parser.add_argument('--output', type=str, help='path to output csv file')
    args = parser.parse_args()
    main(args)
