# ImmunoSeq
Methods to predict antibody immunogenicity

ImmunoSeq -- an interpretable and applicable method for immunogenicity prediction rooted in the biological principle of immune tolerance

### Installation Guide
1. First check that you have installed python packages listed in requirements.txt.
2. Then download paired human and mouse antibody sequence files from Observed Antibody Sequences (https://opig.stats.ox.ac.uk/webapps/oas/oas_paired/) and move them into `data/` folder
3. Run `python prepare.py` to generate k-mer (k=8-12) peptide library for human proteins, oas paired human antibodies, as well as oas paired mouse antibodies

### Benchmark
1. To run ADA correlation benchmark, use `python eval_ada_correlation.py`
2. To run humanness classification benchmark, use `python eval_humanness_classification.py`
3. To benchmark humanness classification on anbativ dataset, run `eval_abnativ.ipynb`
4. To analyze Hu-mAb 25 antibody pairs, use `python eval_humab25.py`
5. To perform sequence immunogenicity optimization, use `python infer.py`

Please address all questions to huangqiaojing@bytedance.com
