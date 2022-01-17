from datasets import get_datasets, synsetid_to_cate
from pprint import pprint
from collections import defaultdict
from model import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn
import jsd

def get_test_loader():
    _, te_dataset = get_datasets()
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=16, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader

def evaluate_gen(model):
    loader = get_test_loader()
    all_sample = []
    all_ref = []
    for data in loader:
        _, te_pc = data['idx'], data['test_points']
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N)
        m, s = data['mean'].float(), data['std'].float()
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    # Save the generative output
    save_dir = os.path.dirname('checkpoints/')
    np.save(os.path.join(save_dir, "model_out_smp.npy"), sample_pcs.cpu().detach().numpy())
    np.save(os.path.join(save_dir, "model_out_ref.npy"), ref_pcs.cpu().detach().numpy())

    # Compute metrics
    results = compute_all(sample_pcs, ref_pcs, 16, accelerated_cd=True)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = jsd.JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)


def main():
    model = PointFlow().cuda()
    #### checkpoint = 
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        evaluate_gen(model)


if __name__ == '__main__':
    main()