from utils import find_mut, match_string, check_response, get_kmers, site_populations
import pickle
import numpy as np
import matplotlib.pyplot as plt
from args import conf

def humab_25(fname, ncase=25):
    with open(fname, 'r') as f:
        lines = f.readlines()
        names, wt_VHs, wt_VLs, mut_VHs, mut_VLs, mutations_VH, mutations_VL, cdrs_VH_index, cdrs_VL_index = [], [], [], [], [], [], [], [], []
        for i in range(ncase):
            name = lines[i*8].strip().split(',')[0]
            names.append(name)
            wt_VH_seq = ''.join(lines[i*8+1].strip().split(',')[1:]).replace('-', '')
            mut_VH_seq = ''.join(lines[i*8+2].strip().split(',')[1:])
            mut_VH_cdr1, mut_VH_cdr2, mut_VH_cdr3 = mut_VH_seq[26:38].replace('-', ''), mut_VH_seq[55:65].replace('-', ''), mut_VH_seq[104:141].replace('-', '')
            mut_VH_seq = mut_VH_seq.replace('-', '')
            wt_VHs.append(wt_VH_seq)
            mut_VHs.append(mut_VH_seq)
            mutation_VH, mutation_VH_type = find_mut(wt_VH_seq, mut_VH_seq)
            mutations_VH.append(mutation_VH)
            VH_inds = []
            for seq in [mut_VH_cdr1, mut_VH_cdr2, mut_VH_cdr3]:
                ind = match_string(mut_VH_seq, seq)
                VH_inds += ind
            cdrs_VH_index.append(VH_inds)
            wt_VL_seq = ''.join(lines[i*8+5].strip().split(',')[1:]).replace('-', '')
            mut_VL_seq = ''.join(lines[i*8+6].strip().split(',')[1:])
            mut_VL_cdr1, mut_VL_cdr2, mut_VL_cdr3 = mut_VL_seq[26:38].replace('-', ''), mut_VL_seq[55:65].replace('-', ''), mut_VL_seq[104:141].replace('-', '')
            mut_VL_seq = mut_VL_seq.replace('-', '')
            wt_VLs.append(wt_VL_seq)
            mut_VLs.append(mut_VL_seq)
            mutation_VL, mutation_VL_type = find_mut(wt_VL_seq, mut_VL_seq)
            mutations_VL.append(mutation_VL)
            VL_inds = []
            for seq in [mut_VL_cdr1, mut_VL_cdr2, mut_VL_cdr3]:
                ind = match_string(mut_VL_seq, seq)
                VL_inds += ind
            cdrs_VL_index.append(VL_inds)
    return names, wt_VHs, wt_VLs, mut_VHs, mut_VLs, mutations_VH, mutations_VL, cdrs_VH_index, cdrs_VL_index

def benchmark_humab_25(conf, fname, ncase=25, plot_wt=True):
    names, wt_VH, wt_VL, mut_VH, mut_VL, mutations_VH, mutations_VL, cdrs_VH_index, cdrs_VL_index = humab_25(fname, ncase)
    pools_id1, pools_id2, pools_id3 = {}, {}, {}
    for i in range(len(names)):
        mut_response_VH, mut_response_VL, mut_scores_VH, mut_scores_VL = 0, 0, {}, {}
        wt_response_VH, wt_response_VL, wt_scores_VH, wt_scores_VL = 0, 0, {}, {}
        mut_total_kmers_VH, mut_total_kmers_VL, wt_total_kmers_VH, wt_total_kmers_VL = 0, 0, 0, 0
        for mer in range(8, 13):
            if mer not in pools_id1.keys():
                pkl = f'data/human_{str(mer)}mer.dump'
                with open(pkl, 'rb') as f:
                    pools = pickle.load(f)
                pools_id1[mer] = pools
            if mer not in pools_id2.keys():
                pkl = f'data/oas_paired_human_{str(mer)}mer.dump'
                with open(pkl, 'rb') as f:
                    pools = pickle.load(f)
                pools_id2[mer] = pools
            if mer not in pools_id3.keys():
                pkl = f'data/oas_paired_mouse_{str(mer)}mer.dump'
                with open(pkl, 'rb') as f:
                    pools = pickle.load(f)
                pools_id3[mer] = pools
            mut_kmers_VH = get_kmers(mut_VH[i], mer, mer)
            mut_kmers_VL = get_kmers(mut_VL[i], mer, mer)
            mut_subscores_VH, mut_subresponse_VH = check_response(mut_kmers_VH, mer, [pools_id1[mer], pools_id2[mer]], [pools_id3], mut_scores_VH)
            mut_subscores_VL, mut_subresponse_VL = check_response(mut_kmers_VL, mer, [pools_id1[mer], pools_id2[mer]], [pools_id3], mut_scores_VL)
            mut_response_VH += mut_subresponse_VH
            mut_response_VL += mut_subresponse_VL
            mut_total_kmers_VH += len(mut_kmers_VH)
            mut_total_kmers_VL += len(mut_kmers_VL)
            mut_scores_VH = mut_subscores_VH
            mut_scores_VL = mut_subscores_VL
            wt_kmers_VH = get_kmers(wt_VH[i], mer, mer)
            wt_kmers_VL = get_kmers(wt_VL[i], mer, mer)
            wt_subscores_VH, wt_subresponse_VH = check_response(wt_kmers_VH, mer, [pools_id1[mer], pools_id2[mer]], [pools_id3], wt_scores_VH)
            wt_subscores_VL, wt_subresponse_VL = check_response(wt_kmers_VL, mer, [pools_id1[mer], pools_id2[mer]], [pools_id3], wt_scores_VL)
            wt_response_VH += wt_subresponse_VH
            wt_response_VL += wt_subresponse_VL
            wt_total_kmers_VH += len(wt_kmers_VH)
            wt_total_kmers_VL += len(wt_kmers_VL)
            wt_scores_VH = wt_subscores_VH
            wt_scores_VL = wt_subscores_VL    
        print(names[i], (mut_response_VH / mut_total_kmers_VH + mut_response_VL / mut_total_kmers_VL) / 2, (wt_response_VH / wt_total_kmers_VH + wt_response_VL / wt_total_kmers_VL) / 2)
        mut_VH_array1, mut_VH_array2 = site_populations(mut_scores_VH)
        mut_VL_array1, mut_VL_array2 = site_populations(mut_scores_VL)
        wt_VH_array1, wt_VH_array2 = site_populations(wt_scores_VH)
        wt_VL_array1, wt_VL_array2 = site_populations(wt_scores_VL)
        ticks1, ticks2 = np.arange(len(mut_VH[i])), np.arange(len(mut_VL[i]))
        # plot population VH
        fig1 = plt.figure(figsize=(15, 9))
        ax1 = fig1.add_subplot(111)
        ax1.plot(mut_VH_array1, mut_VH_array2, color='purple', linewidth=2.5, label='Experimentally Humanized (HUM)', marker='o', markersize=4, markerfacecolor='white')
        if plot_wt:
           ax1.plot(wt_VH_array1, wt_VH_array2, color='orange', linewidth=2.5, label='Precursor (WT)', linestyle='--', marker='o', markersize=4, markerfacecolor='white')
        ax1.text(-0.07, -0.02, "WT - ", ha='right', va='top', color='black', transform=ax1.get_xaxis_transform(), fontsize=10)
        ax1.text(-0.07, -0.065, "HUM - ", ha='right', va='bottom', color='black', transform=ax1.get_xaxis_transform(), fontsize=10)
        for pos, wt, mut in zip(ticks1, list(wt_VH[i]), list(mut_VH[i])):
            if wt != mut:
                ax1.text(pos, -0.02, f"{wt}", ha='center', va='top', color='#d62728', fontweight='bold', transform=ax1.get_xaxis_transform(), fontsize=10)
                ax1.text(pos, -0.065, f"{mut}", ha='center', va='bottom', color='#d62728', fontweight='bold', transform=ax1.get_xaxis_transform(), fontsize=10)
            else:
                ax1.text(pos, -0.02, f"{wt}", ha='center', va='top', color='black', transform=ax1.get_xaxis_transform(), fontsize=10)
                ax1.text(pos, -0.065, f"{mut}", ha='center', va='bottom', color='black', transform=ax1.get_xaxis_transform(), fontsize=10)
        ax1.set_xlim([-2, len(mut_VH[i])])
        ax1.set_ylim([-0.02, 1.02])
        ax1.set_ylabel('Hit Rate', fontsize=14, color='#404040', labelpad=12)
        ax1.set_title(f'VH-{names[i]}', fontsize=14)
        for j in range(3):
            ax1.fill_between([cdrs_VH_index[i][j*2], cdrs_VH_index[i][j*2+1]], 1.02, -0.02, color='green', alpha=0.15)
        for ind in mutations_VH[i]:
            ax1.axvline(ind, color='gray', linestyle='--')
        ax1.legend(loc='lower left', fontsize=10, frameon=False)
        ax1.set_xticks(ticks1)
        ax1.tick_params(axis='x', which='both', length=5, labelbottom=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85)
        plt.savefig(f'plot/{names[i]}_VH.png', dpi=300)
        plt.close()
        # plot population VL
        fig2 = plt.figure(figsize=(15, 9))
        ax2 = fig2.add_subplot(111)
        ax2.plot(mut_VL_array1, mut_VL_array2, color='purple', linewidth=2.5, label='Experimentally Humanized (HUM)', marker='o', markersize=4, markerfacecolor='white')
        if plot_wt:
           ax2.plot(wt_VL_array1, wt_VL_array2, color='orange', linewidth=2.5, label='Precursor (WT)', linestyle='--', marker='o', markersize=4, markerfacecolor='white')
        ax2.text(-0.07, -0.02, "WT - ", ha='right', va='top', color='black', transform=ax2.get_xaxis_transform(), fontsize=10)
        ax2.text(-0.07, -0.065, "HUM - ", ha='right', va='bottom', color='black', transform=ax2.get_xaxis_transform(), fontsize=10)
        for pos, wt, mut in zip(ticks2, list(wt_VL[i]), list(mut_VL[i])):
            if wt != mut:
                ax2.text(pos, -0.02, f"{wt}", ha='center', va='top', color='#d62728', fontweight='bold', transform=ax2.get_xaxis_transform(), fontsize=10)
                ax2.text(pos, -0.065, f"{mut}", ha='center', va='bottom', color='#d62728', fontweight='bold', transform=ax2.get_xaxis_transform(), fontsize=10)
            else: 
                ax2.text(pos, -0.02, f"{wt}", ha='center', va='top', color='black', transform=ax2.get_xaxis_transform(), fontsize=10)
                ax2.text(pos, -0.065, f"{mut}", ha='center', va='bottom', color='black', transform=ax2.get_xaxis_transform(), fontsize=10)
        ax2.set_xlim([-2, len(mut_VL[i])])
        ax2.set_ylim([-0.02, 1.02])
        ax2.set_ylabel('Hit Rate', fontsize=14, color='#404040', labelpad=12)
        ax2.set_title(f'VL-{names[i]}', fontsize=14)
        for j in range(3):
            ax2.fill_between([cdrs_VL_index[i][j*2], cdrs_VL_index[i][j*2+1]], 1.02, -0.02, color='green', alpha=0.15)
        for ind in mutations_VL[i]:
            ax2.axvline(ind, color='gray', linestyle='--')
        ax2.legend(loc='lower left', fontsize=10, frameon=False)
        ax2.set_xticks(ticks2)
        ax2.tick_params(axis='x', which='both', length=5, labelbottom=False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85)
        plt.savefig(f'plot/{names[i]}_VL.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    benchmark_humab_25(conf, 'data/humab_25.csv', ncase=25, plot_wt=True)