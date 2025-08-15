import pickle
import numpy as np
import pandas as pd
from utils import get_kmers, check_response, classification
from args import conf

def benckmark_humanness(conf, pool_pos=[], pool_neg=[], classify=True):
    path = 'data/biophi_humanness.csv'
    df = pd.read_csv(path, index_col=None, delimiter=';')
    names = df.Antibody.tolist()
    seqs = df.Seqs.tolist()
    specy = df.Species.tolist()
    pred, true = [], []
    report = []
    for i in range(len(names)):
        scores, response, total_kmers = {}, 0, 0
        seq = seqs[i].split(",")
        sub_pred = []
        for s in seq:
            response_s, total_kmers_s = 0, 0
            for mer in range(conf.min_mer, conf.max_mer):
                kmers = get_kmers(s, mer, mer)
                pool_mer, neg_pool_mer = [], []
                for item in pool_pos:
                    if mer not in item.keys(): continue
                    pool_mer.append(item[mer])
                for item in pool_neg:
                    if mer not in item.keys(): continue
                    neg_pool_mer.append(item[mer])
                subscores, subresponse = check_response(kmers, mer, pool_mer, neg_pool_mer, scores)
                response += subresponse
                total_kmers += len(kmers)
                scores = subscores
                response_s += subresponse
                total_kmers_s += len(kmers)
            sub_pred.append(response_s / total_kmers_s)
        pred.append(np.mean(sub_pred))
        if specy[i] == "Human":
            true.append(1)
        else:
            true.append(0)
        report.append((response, total_kmers))        
    if classify:
        classification(pred, true)
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
    report = benckmark_humanness(conf, pool_pos=pool_pos, pool_neg=pool_neg, classify=True)
    return report

if __name__ == '__main__':
    main()