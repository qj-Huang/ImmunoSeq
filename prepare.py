import os
import glob
import json
import csv
import gzip 
import pickle
from bloom_filter2 import BloomFilter
from utils import get_kmers

def prepare_human():
    assert os.path.exists('data/HUMAN.json')
    with open('data/HUMAN.json', 'r') as f:
        data = json.load(f)
    return data.keys()

def prepare_oas_mouse():
    seqs = []
    assert os.path.exists('data/OAS_PAIRED_MOUSE')
    for path in glob.glob(f'data/OAS_PAIRED_MOUSE/*.csv.gz'):
        f = gzip.open(path, 'rt')
        csv_reader = csv.reader(f)
        meta = next(csv_reader)
        meta = json.loads(meta[0])
        header = next(csv_reader)
        index_map = {key: i for i, key in enumerate(header) if key in ['sequence_alignment_aa_heavy', 'sequence_alignment_aa_light']}
        for line in csv_reader:
            for _, idx in index_map.items():
                seqs.append(line[idx])
    return seqs

def prepare_oas_human():
    seqs = []
    assert os.path.exists('data/OAS_PAIRED_HUMAN')
    for path in glob.glob(f'data/OAS_PAIRED_HUMAN/*.csv.gz'):
        f = gzip.open(path, 'rt')
        csv_reader = csv.reader(f)
        meta = next(csv_reader)
        meta = json.loads(meta[0])
        if meta['Species'] != 'human':continue
        if meta['Disease'] != 'None':continue
        header = next(csv_reader)
        index_map = {key: i for i, key in enumerate(header) if key in ['sequence_alignment_aa_heavy', 'sequence_alignment_aa_light']}
        for line in csv_reader:
            for _, idx in index_map.items():
                seqs.append(line[idx])
    return seqs

def dump_seqs(seqs, database):
    for mer in range(8, 13):
        datas = []
        for seq in seqs:
            kmer = get_kmers(seq, mer, mer)
            datas += kmer
        datas = list(set(datas))
        pkl = f'data/{database}_{str(mer)}mer.dump'
        if not os.path.exists(pkl):
            pools = BloomFilter(max_elements=1.5e8)
            for item in datas:
                pools.add(item)
            with open(pkl, 'wb') as f:
                pickle.dump(pools, f)

if __name__ == '__main__':
    human_protein_seqs = prepare_human()
    oas_paired_mouse_seqs = prepare_oas_mouse()
    oas_paired_human_seqs = prepare_oas_human()
    for seqs, database in zip([human_protein_seqs, oas_paired_mouse_seqs, oas_paired_human_seqs], ['human', 'oas_paired_mouse', 'oas_paired_human']):
        dump_seqs(seqs, database)