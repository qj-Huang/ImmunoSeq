import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns

def get_kmers(sequence, min_k, max_k):
    kmers = []
    for k in range(min_k, max_k + 1):
        for i in range(0, len(sequence) - k + 1):
            kmers.append(sequence[i:i + k])
    return kmers    

def generate_single_mutation(protein_sequence, annotation='', fixed=[]): # fixed = [27-31,32,65-68]
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    mutations = {}
    fixed_region = []
    if len(fixed) != 0:
        for item in fixed:
            if '-' not in item:
                fixed_region.append(item)
            else:
                left, right = item.split("-")[0], item.split("-")[1]
                for i in range(int(left), int(right) + 1):
                    fixed_region.append(i)
    for i in range(len(protein_sequence)):
        if i in fixed_region: continue
        for aa in amino_acids:
            if aa!= protein_sequence[i]:
                mutation = protein_sequence[:i] + aa + protein_sequence[i+1:]
                symbol = annotation + protein_sequence[i] + str(i+1) + aa
                mutations[symbol] = mutation
    return mutations

def check_response(seq_pools, mer, dat_pool_pos=[], dat_pool_neg=[], update_dict={}):
    responses = 0
    for item in seq_pools:
        ind = seq_pools.index(item)
        for resid in range(ind, ind + mer):
            if resid not in update_dict.keys():
                update_dict[resid] = 0
        flag = False
        for dat in dat_pool_pos:
            if item in dat:
                flag = True
            if flag:            
                responses += 1
                for resid in range(ind, ind + mer):
                    update_dict[resid] += 1
                break
        for dat in dat_pool_neg:
            if item in dat:
                responses -= 0.2
                for resid in range(ind, ind + mer):
                    update_dict[resid] -= 0.2
                break
    return update_dict, responses

def count_segments(nseq):
    counts = []
    for i in range(nseq):
        total = 0
        for k in range(8, 13):
            start_min = max(0, i - k + 1)
            start_max = min(i, nseq - k)
            if start_max >= start_min:
                cnt = start_max - start_min + 1
            else:
                cnt = 0
            total += cnt
        counts.append(total)
    return counts

def site_populations(dict_score):
    score_key = sorted(dict_score.keys())
    score_value = [dict_score[key] for key in score_key]
    total_score = count_segments(len(score_key))
    score_value = [value / total_score[i] for i, value in enumerate(score_value)]
    return score_key, score_value

def classification(pred, label, nfloat=4):
    intervals = np.arange(0, max(pred), 0.001)
    auroc = roc_auc_score(label, pred)
    auprc = average_precision_score(label, pred)  
    print('interval,mcc,auprc,auroc,accuracy,f1_score,recall,precision')
    for interval in intervals:
        fpred = [1 if i >= interval else 0 for i in pred]
        metric = emit_metrics(fpred, label)
        metric = map(lambda x: round(x, nfloat), metric)
        accuracy, f1_score, mcc, recall, precision = metric
        print(f"{round(interval, nfloat)},{mcc},{auprc},{auroc},{accuracy},{f1_score},{recall},{precision}")

def emit_metrics(pre, lab):
    pre = torch.tensor(pre, dtype=float)
    lab = torch.tensor(lab, dtype=int)
    tp = ((pre > 0.5) & (lab > 0.5)).sum()
    tn = ((pre < 0.5) & (lab < 0.5)).sum()
    fp = ((pre > 0.5) & (lab < 0.5)).sum()
    fn = ((pre < 0.5) & (lab > 0.5)).sum()
    accuracy = (tp + tn) / pre.shape[0]
    recall = tp / (lab > 0.5).sum()
    precision = tp / (pre > 0.5).sum()
    f1_score = 2 * precision * recall / (precision + recall)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denominator.cpu().numpy())
    return accuracy.item(), f1_score.item(), mcc.item(), recall.item(), precision.item()

def find_mut(seq1, seq2):
    mut_pos = []
    mut_type = []
    assert len(seq1) == len(seq2)
    for i in range(len(seq2)):
        if seq1[i] != seq2[i]:
            mut_pos.append(i)
            mut_type.append(f"{seq1[i]}{i+1}{seq2[i]}")
    return mut_pos, mut_type

def match_string(text, pattern):
    match = re.search(pattern, text)
    return [match.start(), match.end()]

def plot_ada(VH_score, VL_score, dirname, species):
    VH_array1, VH_array2 = site_populations(VH_score, normalized=True)
    VL_array1, VL_array2 = site_populations(VL_score, normalized=True)
    scatter_colors = sns.color_palette("husl", n_colors=5) 
    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(211)
    ax1.bar(VH_array1, VH_array2, width=0.2, linewidth=5, color=scatter_colors[-1])
    ax1.plot(VH_array1, VH_array2, color=scatter_colors[-1], linewidth=2, marker='o', markersize=4, markerfacecolor='white')
    ax1.set_ylabel('VH Hit Rate', fontsize=12)
    ax1.set_xlim(-1, len(VH_array1))
    ax1.set_title(dirname, fontsize=14)
    ax1.axhline(y=0, color='black', linestyle='--')
    ax2 = fig.add_subplot(212)
    ax2.bar(VL_array1, VL_array2, width=0.2, linewidth=5, color=scatter_colors[-1])
    ax2.plot(VL_array1, VL_array2, color=scatter_colors[-1], linewidth=2, marker='o', markersize=4, markerfacecolor='white')
    ax2.set_xlabel('Residue Position', fontsize=12)
    ax2.set_xlim(-1, len(VL_array1))
    ax2.set_ylabel('VL Hit Rate', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='--')
    for ax in [ax1, ax2]:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#808080')
            ax.spines['bottom'].set_color('#808080')
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(f'plot/{dirname}_{species}_site.png', dpi=600)
    plt.close()