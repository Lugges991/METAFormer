import pandas as pd
import os
from tqdm import tqdm
import wget


# base_str = https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/[strategy]/[derivative]/[file identifier]_[derivative].[ext]


# https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_aal/KKI_0050822_rois_aal.1D
base_str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/[filt]/[roi]/[file identifier]_[roi].1D"


roi_str = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/[pipeline]/filt_global/[derivative]/[file identifier]_[derivative].1D"


def create_url(pipe, roi, fg):
    url = base_str.replace("[pipeline]", pipe)
    url = url.replace("[filt]", fg)
    url = url.replace("[roi]", roi)
    return url


def download_abide1_roi(pheno_file, out_dir, pipe, roi, fg):
    df = pd.read_csv(pheno_file)

    url = create_url(pipe, roi, fg)

    # create out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        sub_id = row["SUB_ID"]
        site = row["SITE_ID"]
        url = url.replace("[file identifier]", site + "_00" + str(sub_id))
        out_file = f"{out_dir}/{site}_{sub_id}_{roi}.1D"
        try:
            wget.download(url, out_file)
        except Exception as e:
            print(e)
            print(f"Failed to download {url} to {out_file}")


def download_abide1_pcp(pheno_file, out_dir):
    df = pd.read_csv(pheno_file)

    # create out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, row in tqdm(df.iterrows(), total=len(df)):
        sub_id = row["SUB_ID"]
        site = row["SITE_ID"]
        url = base_str.replace("[file identifier]", site + "_00" + str(sub_id))
        out_file = f"{out_dir}/{site}_{sub_id}_func_preproc.nii.gz"
        try:
            wget.download(url, out_file)
        except Exception as e:
            print(e)
            print(f"Failed to download {url} to {out_file}")


if __name__ == "__main__":
    # argument parser
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pheno_file", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--pipe", type=str, default="dparsf")
    parser.add_argument("--roi", type=str, default="cc200")
    parser.add_argument("--fg", type=str, default="filt_noglobal")
    args = parser.parse_args()

    download_abide1_roi(
        args.pheno_file,
        args.out_dir,
        args.pipe,
        args.roi,
        args.fg)
