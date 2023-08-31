import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import captum
import argparse

from captum.attr import IntegratedGradients, DeepLift, Saliency, FeatureAblation, GradientShap, KernelShap, DeepLiftShap
from captum.metrics import infidelity, sensitivity_max

from METAFormer.models import METAFormer
from METAFormer.dataloader import MultiAtlas

torch.manual_seed(1337)
np.random.seed(1337)

def perturb_func(input):
    aal, cc200, dos160 = input
    noise_aal = torch.tensor(np.random.normal(0, 0.003, aal.shape)).float()
    noise_cc200 = torch.tensor(np.random.normal(0, 0.003, cc200.shape)).float()
    noise_dos160 = torch.tensor(np.random.normal(0,0.003, dos160.shape)).float()
    return (noise_aal, noise_cc200, noise_dos160), (aal - noise_aal, cc200 - noise_cc200, dos160 - noise_dos160)


def get_deep_lift_sens(model, x_list, targets, baselines):
    dl = DeepLift(model)
    sens = sensitivity_max(dl.attribute, x_list, target=targets, baselines=baselines)
    return sens

def get_integrated_gradients_sens(model, x_list, targets, baselines):
    ig = IntegratedGradients(model)
    sens = sensitivity_max(ig.attribute, x_list, target=targets, baselines=baselines)
    return sens

def get_feature_ablation_sens(model, x_list, targets, baselines):
    fa = FeatureAblation(model)
    sens = sensitivity_max(fa.attribute, x_list, target=targets, baselines=baselines)
    return sens

def get_shap_sens(model, x_list,baselines, targets):
    gs = GradientShap(model)
    sens = sensitivity_max(gs.attribute, x_list, baselines=baselines, target=targets)
    return sens

def get_saliency_sens(model, x_list, targets, baselines):
    sal = Saliency(model)
    sens = sensitivity_max(sal.attribute, x_list, target=targets, baselines=baselines)
    return sens

def get_kernel_shap_sens(model, x_list, baselines, targets):
    ks = KernelShap(model)
    sens = sensitivity_max(ks.attribute, x_list, baselines=baselines, target=targets)
    return sens

def get_deep_lift_shap(model, x_list, baselines, targets):
    dls = DeepLiftShap(model)
    sens = sensitivity_max(dls.attribute, x_list, baselines=baselines, target=targets)
    return sens


def main(args):
    model = METAFormer()
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    test_df = pd.read_csv(args.test_csv)

    ds = MultiAtlas(test_df)
    dl = DataLoader(ds, batch_size=256, shuffle=False)

    # get batch, note batch_size is so large that all data is in one batch
    x_batch, y_batch = next(iter(dl))

    (x_aal, x_cc200, x_dos160) = x_batch[0], x_batch[1], x_batch[2]
    y_batch = y_batch

    target = y_batch.reshape(-1).to(torch.int64)

    model.cpu()
    model.eval()

    inf_df = calc_infidelity(model, x_aal, x_cc200, x_dos160, target)
    inf_df.to_csv("infidelity.csv")
    sens_df = calc_sensitivity(model, x_aal, x_cc200, x_dos160, target)
    sens_df.to_csv("sensitivity.csv")

    sal = Saliency(model)
    sal_sens = sensitivity_max(sal.attribute, (x_aal, x_cc200, x_dos160), target=target)
    print("Saliency Sensitivity: ", sal_sens)
    sal_attr = sal.attribute((x_aal, x_cc200, x_dos160), target=target)
    sal_attr = sal_attr.detach().numpy()
    sal_inf = infidelity(model, perturb_func, (x_aal, x_cc200, x_dos160),(sal_attr[0], sal_attr[1], sal_attr[2]), target=target)
    print("Saliency Infidelity: ", sal_inf.item())


def calc_sensitivity(model, x_aal, x_cc200, x_dos160, target):
    results = []

    for i in range(-10, 11, 1):
        print("Baseline: ", i/10)
        baseline_aal = torch.ones_like(x_aal)*(i/10)
        baseline_cc200 = torch.ones_like(x_cc200)*(i/10)
        baseline_dos160 = torch.ones_like(x_dos160)*(i/10)
        ig_sens = get_integrated_gradients_sens(model, (x_aal, x_cc200, x_dos160), target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))
        dl_sens = get_deep_lift_sens(model, (x_aal, x_cc200, x_dos160), target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))
        fa_sens = get_feature_ablation_sens(model, (x_aal, x_cc200, x_dos160), target, baselines=i/10)
        gs_sens = get_shap_sens(model, (x_aal, x_cc200, x_dos160), (baseline_aal, baseline_cc200, baseline_dos160), target)
        dls_sens = get_deep_lift_sens(model, (x_aal, x_cc200, x_dos160), target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))

        # save mean and std for each method

        results_dict = {
            "Baseline": i/10,
            "ig_mean": ig_sens[0].item(),
            "ig_std": ig_sens[1].item(),
            "dl_mean": dl_sens[0].item(),
            "dl_std": dl_sens[1].item(),
            "fa_mean": fa_sens[0].item(),
            "fa_std": fa_sens[1].item(),
            "gs_mean": gs_sens[0].item(),
            "gs_std": gs_sens[1].item(),
            "dls_mean": dls_sens[0].item(),
            "dls_std": dls_sens[1].item()
        }

        results.append(results_dict)

    results_df = pd.DataFrame(results)

    return results_df



def calc_infidelity(model, x_aal, x_cc200, x_dos160, target):

    results = []

    saliency = Saliency(model)
    dl = DeepLift(model)
    ig = IntegratedGradients(model)
    fa = FeatureAblation(model)
    shap = GradientShap(model)
    dls = DeepLiftShap(model)

    for i in range(-10, 11, 1):
        print("Baseline: ", i/10)
        baseline_aal = torch.ones_like(x_aal)*(i/10)
        baseline_cc200 = torch.ones_like(x_cc200)*(i/10)
        baseline_dos160 = torch.ones_like(x_dos160)*(i/10)

        dl_attr = dl.attribute((x_aal, x_cc200, x_dos160), target=target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))
        ig_attr = ig.attribute((x_aal, x_cc200, x_dos160), target=target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))
        fa_attr = fa.attribute((x_aal, x_cc200, x_dos160), target=target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))
        shap_attr = shap.attribute((x_aal, x_cc200, x_dos160), target=target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))
        dls_attr = dls.attribute((x_aal, x_cc200, x_dos160), target=target, baselines=(baseline_aal, baseline_cc200, baseline_dos160))

        fid_dl = infidelity(model, perturb_func, (x_aal, x_cc200, x_dos160),(dl_attr[0], dl_attr[1], dl_attr[2]), target=target)
        fid_ig = infidelity(model, perturb_func, (x_aal, x_cc200, x_dos160),(ig_attr[0], ig_attr[1], ig_attr[2]), target=target)
        fid_fa = infidelity(model, perturb_func, (x_aal, x_cc200, x_dos160), (fa_attr[0], fa_attr[1], fa_attr[2]), target=target)
        fid_shap = infidelity(model, perturb_func, (x_aal, x_cc200, x_dos160), (shap_attr[0], shap_attr[1], shap_attr[2]), target=target)
        fid_dls = infidelity(model, perturb_func, (x_aal, x_cc200, x_dos160), (dls_attr[0], dls_attr[1], dls_attr[2]), target=target)


        results_dict = {
            "baseline": i/10,
            "dl_mean": fid_dl.mean().item(),
            "dl_std": fid_dl.std().item(),
            "ig_mean": fid_ig.mean().item(),
            "ig_std": fid_ig.std().item(),
            "dls_mean": fid_dls.mean().item(),
            "dls_std": fid_dls.std().item(),
            "shap_mean": fid_shap.mean().item(),
            "shap_std": fid_shap.std().item(),
            "fa_mean": fid_fa.mean().item(),
            "fa_std": fid_fa.std().item()
        }

        results.append(results_dict)

    results_df = pd.DataFrame(results)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)

    args = parser.parse_args()
    main(args)

