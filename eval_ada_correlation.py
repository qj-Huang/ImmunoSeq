import torch
import pickle
import pandas as pd
import torchmetrics.functional as metrics
from utils import get_kmers, check_response, plot_ada
from args import conf

def benckmark_ada(conf, pool_pos=[], pool_neg=[], origin=['Human', 'Humanized', 'Humanized/Chimeric', 'Chimeric', 'Mouse'], plot=False):
    path = 'data/ADA_Clinical_Ab_2021.csv'
    df = pd.read_csv(path, index_col=None)
    df = df[df['Species'].isin(origin)]
    names = df.Antibody.tolist()
    seqs_VH = df.VH.tolist()
    seqs_VL = df.VL.tolist()
    ada = df.Immunogenicity.tolist()
    specy = df.Species.tolist()
    report = []
    pred, true = [], []
    for i in range(len(names)):
        scores_VH, scores_VL, response_VH, response_VL, total_kmers_VH, total_kmers_VL = {}, {}, 0, 0, 0, 0
        for mer in range(conf.min_mer, conf.max_mer):
            kmers_VH = get_kmers(seqs_VH[i], mer, mer)
            kmers_VL = get_kmers(seqs_VL[i], mer, mer)
            pos_pool_mer, neg_pool_mer = [], []
            for item in pool_pos:
                if mer not in item.keys(): continue
                pos_pool_mer.append(item[mer])
            for item in pool_neg:
                if mer not in item.keys(): continue
                neg_pool_mer.append(item[mer])
            subscores_VH, subresponse_VH = check_response(kmers_VH, mer, pos_pool_mer, neg_pool_mer, scores_VH)
            subscores_VL, subresponse_VL = check_response(kmers_VL, mer, pos_pool_mer, neg_pool_mer, scores_VL)
            response_VH += subresponse_VH
            response_VL += subresponse_VL
            total_kmers_VH += len(kmers_VH)
            total_kmers_VL += len(kmers_VL)
            scores_VH = subscores_VH
            scores_VL = subscores_VL
        if plot:
            plot_ada(scores_VH, scores_VL, names[i], specy[i].replace('/', '_'))
        pred.append((response_VH / total_kmers_VH + response_VL / total_kmers_VL) / 2)
        true.append(ada[i])
        report.append((response_VH, response_VL, total_kmers_VH, total_kmers_VL))
    pred = torch.tensor(pred, dtype=float)
    true = torch.tensor(true, dtype=float)
    print(f"{metrics.pearson_corrcoef(pred, true).item():.4f}", f"{metrics.spearman_corrcoef(pred, true).item():.4f}")
    return report

def main():
    data_human, data_oas_pair_human, data_oas_pair_mouse = {}, {}, {}
    for mer in range(conf.min_mer, conf.max_mer):
        if mer not in data_oas_pair_human.keys():
            pkl = f'data/oas_paired_human_{str(mer)}mer.dump'
            with open(pkl, 'rb') as f:
                pools = pickle.load(f)
            data_oas_pair_human[mer] = pools
        if mer not in data_human.keys():
            pkl = f'data/human_{str(mer)}mer.dump'
            with open(pkl, 'rb') as f:
                pools = pickle.load(f)
            data_human[mer] = pools
        if mer not in data_oas_pair_mouse.keys():
            pkl = f'data/oas_paired_mouse_{str(mer)}mer.dump'
            with open(pkl, 'rb') as f:
                pools = pickle.load(f)
            data_oas_pair_mouse[mer] = pools
    pool_pos = [data_human, data_oas_pair_human]
    pool_neg = [data_oas_pair_mouse]
    report = benckmark_ada(conf, pool_pos=pool_pos, pool_neg=pool_neg)
    return report

if __name__ == '__main__':
    main()