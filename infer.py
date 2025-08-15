from utils import generate_single_mutation, get_kmers, check_response
import pandas as pd
import numpy as np
import pickle
from args import conf

def infer_mut(seqs, name, fixed_VH=[], fixed_VL=[], topk=1):  # fixed = [27-31,32,65-68]
    pools_human, pools_oas_pair_human, pools_oas_pair_mouse = {}, {}, {}
    pred, mut, top_seq = [], [], []
    seqs = seqs.split(",")
    single_mut_vh = generate_single_mutation(seqs[0], annotation="VH_", fixed=fixed_VH)
    single_mut_vl = generate_single_mutation(seqs[1], annotation="VL_", fixed=fixed_VL)
    for k, v in single_mut_vh.items():
        single_mut_vh[k] = f"{v},{seqs[1]}"
    for k, v in single_mut_vl.items():
        single_mut_vl[k] = f"{seqs[0]},{v}"
    single_mut = {**single_mut_vh, **single_mut_vl, 'wt':f"{seqs[0]},{seqs[1]}"}
    for k, va in single_mut.items():
        scores, response, total_kmers = {}, 0, 0
        v = va.split(',')
        sub_pred = []
        for s in v:
            response_s, total_kmers_s = 0, 0
            for mer in range(conf.min_mer, conf.max_mer):
                if mer not in pools_human.keys():
                    pkl = f'data/human_{str(mer)}mer.dump'
                    with open(pkl, 'rb') as f:
                        pools = pickle.load(f)
                    pools_human[mer] = pools
                if mer not in pools_oas_pair_human.keys():
                    pkl = f'data/oas_paired_human_{str(mer)}mer.dump'
                    with open(pkl, 'rb') as f:
                        pools = pickle.load(f)
                    pools_oas_pair_human[mer] = pools
                if mer not in pools_oas_pair_mouse.keys():
                    pkl = f'data/oas_paired_mouse_{str(mer)}mer.dump'
                    with open(pkl, 'rb') as f:
                        pools = pickle.load(f)
                    pools_oas_pair_mouse[mer] = pools
                kmers = get_kmers(s, mer, mer)
                subscores, subresponse = check_response(kmers, mer, [pools_human[mer], pools_oas_pair_human[mer]], [pools_oas_pair_mouse[mer]], scores)
                response += subresponse
                scores = subscores
                total_kmers += len(kmers)
                response_s += subresponse
                total_kmers_s += len(kmers)
            sub_pred.append(response_s / total_kmers_s)
        pred.append(np.mean(sub_pred))
        mut.append(k)
        top_seq.append(va)
    result = {
        'mut':mut,
        'score':pred,
        'seq':top_seq
    }
    df = pd.DataFrame(result)
    df_sort = df.sort_values('score', ascending=False)
    if topk > 0:
        topk_seq = df_sort.head(topk)
        return topk_seq.to_dict(orient='records')
    else:
        df_sort.to_csv(f'design/{name}_infer_mut_oneshot.csv', index=None)

def predict_ada(seq, pool_pos=[], pool_neg=[]):
    scores, response, total_kmers = {}, 0, 0
    seq = seq.split(",")
    for s in seq:
        for mer in range(conf.min_mer, conf.max_mer):
            kmers = get_kmers(s, mer, mer)
            pool_mer_pos, neg_pool_mer = [], []
            for item in pool_pos:
                pool_mer_pos.append(item[mer])
            for item in pool_neg:
                neg_pool_mer.append(item[mer])
            subscores, subresponse = check_response(kmers, mer, pool_mer_pos, neg_pool_mer, scores)
            response += subresponse
            total_kmers += len(kmers)
            scores = subscores
    return response / total_kmers

if __name__ == '__main__':
    # science cov19 nanobody https://www.science.org/doi/10.1126/science.abe4747； https://www.sciencedirect.com/science/article/pii/S0969212621004184
    seqs = {
        "nb9": "HVQLVESGGGLVQTGGSLRLSCAFSGYTFSTFPTAWFRQAPGKEREFVAGIRWNGGSRDYTEYADFVKGRFTISRDNAKNMIYLQMISLKPEDTALYYCAASHGVVDGTSVNGYRYWGQGTQVTVSS",
        "nb9_mut": "EVQLVESGGGLVQPGGSLRLSCAFSGYTFSTFPTAWFRQAPGKGREFVAGIRWNGGSRDYTEYADFVKGRFTISRDNAKNTLYLQMNSLKAEDTALYYCAASHGVVDGTSVNGYRYWGQGTLVTVSS",
        "nb17": "HVQLVESGGGLVQAGGSLRLSCAASGSIFSSNAMSWYRQAPGKQRELVASITSGGNADYADSVKGRFTISRDKNTVYPEMSSLKPADTAVYYCHAVGQEASAYAPRAYWGQGTQVTVSS",
        "nb17_mut": "EVQLVESGGGLVQPGGSLRLSCAASGSIFSSNAMSWYRQAPGKGREWVASITSGGNADYADSVKGRFTISRDKNTLYLQMNSLRAEDTAVYYCHAVGQEASAYAPRAYWGQGTLVTVSS",
        "nb34": "DVQLVESGGGLVQAGGSLRLSCAASGFTFSNYVMYWGRQAPGKGREWVSGIDSDGSDTAYASSVKGRFTISRDNAKNTLYLQMNNLKPEDTALYYCVKSKDPYGSPWTRSEFDDYWGQGTQVTVSS",
        "nb34_mut": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYVMYWVRQAPGKGREWVSGIDSDGSDTAYADSVKGRFTISRDNAKNTLYLQMNSLKAEDTALYYCVKSKDPYGSPWTRSEFDDYWGQGTLVTVSS",
        "nb36": "HVQLVESGGGLVQAGGSLTLTCAASGRTFSSETMDMGWFRQAPGKEREFVAADSWNDGSTYYADSVKGRFTISRDSAKNTLYLQMNSLKPEDTAVYYCAAETYSIYEKDDSWGYWGQGTQVTVSS",
        "nb36_mut": "EVQLVESGGGLVQPGGSLRLSCAASGRTFSSETMDMGWFRQAPGKGREFVAADSWNDGSTYYADSVKGRFTISRDNAKNTLYLQMNSLKAEDTAVYYCAAETYSIYEKDDSWGYWGQGTLVTVSS",
        "nb64": "QVQLVESGGGLVQAGGSLRLSCAVSGRTFSIAGMGWFRQAPGKDREFLGGITWNDGTTWYADSVKGRFTISRDNAKNMLSLRMNSLKPEDTAVYYCAAGPRLGSTPRAYDYWGQGTQVTVSS",
        "nb64_mut": "QVQLVESGGGLVQPGGSLRLSCAASGRTFSIAGMGWFRQAPGKGREFVSGITWNDGTTWYADSVKGRFTISRDNAKNTLYLQMNSLKAEDTAVYYCAAGPRLGSTPRAYDYWGQGTLVTVSS",
        "nb93": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGREWVAGITPGSGTFYADSVKGRFTISRDNAKNTLSLEINSLKPEDTALYYCAKCRQEFSWDFSSRDPDDFDYWGQGTQVTVSS",
        "nb93_mut": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGREWVAGITPGSGTFYADSVKGRFTISRDNAKNTLYLEMNSLKAEDTALYYCAKCRQEFSWDFSSRDPDDFDYWGQGTLVTVSS",
        "nb95": "QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYSMGWFRQAQGKEREFVATINGNGRDTYYTNSVKGRFTISRDDATNTVYLQMNSLKPEDTAIYYCAADKDVYYGYTSFPNEYEYWGQGTQVTVSS",
        "nb95_mut": "QVQLVESGGGLVQPGGSLRLSCAASGRTFSSYSMGWFRQAPGKEREFVATINGNGRDTYYADSVKGRFTISRDDAKNTLYLQMNSLKAEDTAVYYCAADKDVYYGYTSFPNEYEYWGQGTLVTVSS",
        "nb105": "HVQLVESGGGLVQAGGSLRLSCAVSGRTFSTYGMAWFRQAPGKERDFVATITRSGETTLYADSVKGRFTISRDNAKNTVYLQMNSLKIEDTAVYYCAVRRDSSWGYSRDLFEYDYWGQGTQVTVSS",
        "nb105_mut": "EVQLVESGGGLVQPGGSLRLSCAASGRTFSTYGMAWFRQAPGKGRDFVATITRSGETTLYADSVKGRFTISRDNAKNTLYLQMNSLKAEDTAVYYCAVRRDSSWGYSRDLFEYDYWGQGTLVTVSS",    
        }
    #humab 25 antibodies
    seqs = {
        "VH_CD28":"EVKLQQSGPGLVTPSQSLSITCTVSGFSLSDYGVHWVRQSPGQGLEWLGVIWAGGGTNYNSALMSRKSISKDNSKSQVFLKMNSLQADDTAVYYCARDKGYSYYYSMDYWGQGTSVTVSS",
        "VH_Campath":"EVKLLESGGGLVQPGGSMRLSCAGSGFTFTDFYMNWIRQPAGKAPEWLGFIRDKAKGYTTEYNPSVKGRFTISRDNTQNMLYLQMNTLRAEDTATYYCAREGHTAAPFDYWGQGVMVTVSS",
        "VH_Bevacizumab":"EIQLVQSGPELKQPGETVRISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYAADFKRRFTFSLETSASTAYLQISNLKNDDTATYFCAKYPHYYGSSHWYFDVWGAGTTVTVSS",
        "VH_Herceptin":"QVQLQQSGPELVKPGASLKLSCTASGFNIKDTYIHWVKQRPEQGLEWIGRIYPTNGYTRYDPKFQDKATITADTSSNTAYLQVSRLTSEDTAVYYCSRWGGDGFYAMDYWGQGASVTVSS",
        "VH_Omalizumab":"DVQLQESGPGLVKPSQSLSLACSVTGYSITSGYSWNWIRQFPGNKLEWMGSITYDGSSNYNPSLKNRISVTRDTSQNQFFLKLNSATAEDTATYYCARGSHYFGHWHFAVWGAGTTVTVSS",
        "VH_Eculizumab":"QVQLQQSGAELMKPGASVKMSCKATGYIFSNYWIQWIKQRPGHGLEWIGEILPGSGSTEYTENFKDKAAFTADTSSNTAYMQLSSLTSEDSAVYYCARYFFGSSPNWYFDVWGAGTTVTVSS",
        "VH_Tocilizumab":"DVQLQESGPVLVKPSQSLSLTCTVTGYSITSDHAWSWIRQFPGNKLEWMGYISYSGITTYNPSLKSRISITRDTSKNQFFLQLNSVTTGDTSTYYCARSLARTTAMDYWGQGTSVTVSS",
        "VH_Pembrolizumab":"QVQLQQPGAELVKPGTSVKLSCKASGYTFTNYYMYWVKQRPGQGLEWIGGINPSNGGTNFNEKFKNKATLTVDSSSSTTYMQLSSLTSEDSAVYYCTRRDYRFDMGFDYWGQGTTLTVSS",
        "VH_Pertuzumab":"EVQLQQSGPELVKPGTSVKISCKASGFTFTDYTMDWVKQSHGKSLEWIGDVNPNSGGSIYNQRFKGKASLTVDRSSRIVYMELRSLTFEDTAVYYCARNLGPSFYFDYWGQGTTLTVSS",
        "VH_Ixekizumab":"QVQLQQSRPELVKPGASVKISCKASGYSFTDYNMNWVKQSNGKSLEWIGVINPNYGTTDYNQRFKGKATLTVDQSSRTAYMQLNSLTSEDSAVYYCVIYDYATGTGGYWGQGSPLTVSS",
        "VH_Palivizumab":"QVELQESGPGILQPSQTLSLTCSFSGFSLSTSGMSVGWIRQPSGEGLEWLADIWWDDKKDYNPSLKSRLTISKDTSSNQVFLKITGVDTADTATYYCARSMITNWYFDVWGAGTTVTVSS",
        "VH_Certolizumab":"QIQLVQSGPELKKPGETVKISCKASGYVFTDYGMNWVKQAPGKAFKWMGWINTYIGEPIYVDDFKGRFAFSLETSASTAFLQINNLKNEDTATYFCARGYRSYAMDYWGQGTSVTVSS",
        "VH_Idarucizumab":"QVQLEQSGPGLVAPSQRLSITCTVSGFSLTSYIVDWVRQSPGKGLEWLGVIWAGGSTGYNSALRSRLSITKSNSKSQVFLQMNSLQTDDTAIYYCASAAYYSYYNYDGFAYWGQGTLVTVSA",
        "VH_Reslizumab":"EVKLLESGGGLVQPSQTLSLTCTVSGLSLTSNSVNWIRQPPGKGLEWMGLIWSNGDTDYNSAIKSRLSISRDTSKSQVFLKMNSLQSEDTAMYFCAREYYGYFDYWGQGVMVTVSS",
        "VH_Solanezumab":"EVKLVESGGGLVQPGGSLKLSCAVSGFTFSRYSMSWVRQTPEKRLELVAQINSVGNSTYYPDTVKGRFTISRDNAEYTLSLQMSGLRSDDTATYYCASGDYWGQGTTLTVSS",
        "VH_Lorvotuzumab":"DVQLVESGGGLVQPGGSRKLSCAASGFTFSSFGMHWVRQAPEKGLEWVAYISSGSFTIYHADTVKGRFTISRDNPKNTLFLQMTSLRAEDTAHYYCARMRKGYAMDYWGQGTTVTVSS",
        "VH_Pinatuzumab":"QVQLQQSGPELVKPGASVKISCKASGYEFSRSWMNWVKQRPGQGREWIGRIYPGDGDTNYSGKFKGKATLTADKSSSTAYMQLSSLTSVDSAVYFCARDGSSWDWYFDVWGAGTTVTVSS",
        "VH_Etaracizumab":"EVQLEESGGGLVKPGGSLKLSCAASGFAFSSYDMSWVRQIPEKRLEWVAKVSSGGGSTYYLDTVQGRFTISRDNAKNTLYLQMSSLNSEDTAMYYCARHNYGSFAYWGQGTLVTVSA",
        "VH_Talacotuzumab":"EVQLQQSGPELVKPGASVKMSCKASGYTFTDYYMKWVKQSHGKSLEWIGDIIPSNGATFYNQKFKGKATLTVDRSSSTAYMHLNSLTSEDSAVYYCTRSHLLRASWFAYWGQGTLVTVSA",
        "VH_Rovalpituzumab":"QIQLVQSGPELKKPGETVKISCKASGYTFTNYGMNWVKQAPGKGLKWMAWINTYTGEPTYADDFKGRFAFSLETSASTASLQIINLKNEDTATYFCARIGDSSPSDYWGQGTTLTVSS",
        "VH_Clazakizumab":"QVSLEESGGRLVTPGTPLTLTCTASGFSLSNYYVTWVRQAPGKGLEWIGIIYGSDETAYATWAIGRFTISKTSTKNTVDLKMTSLTAADTATYFCARDDSSDWDAKFNLWGQGTLVTVSS",
        "VH_Ligelizumab":"QVQLQQSGAELMKPGASVKISCKTTGYTFSMYWLEWVKQRPGHGLEWVGEISPGTFTTNYNEKFKAKATFTADTSSNTAYLQLSGLTSEDSAVYFCARFSHFSGSNYDYFDYWGQGTSLTVSS",
        "VH_Crizanlizumab":"QVQLQQSGPELVKPGALVKISCKASGYTFTSYDINWVKQRPGQGLEWIGWIYPGDGSIKYNEKFKGKATLTVDKSSSTAYMQVSSLTSENSAVYFCARRGEYGNYEGAMDYWGQGTTVTVSS",
        "VH_Mogamulizumab":"EVQLVESGGDLMKPGGSLKISCAASGFIFSNYGMSWVRQTPDMRLEWVATISSASTYSYYPDSVKGRFTISRDNAENSLYLQMNSLRSEDTGIYYCGRHSDGNFAFGYWGRGTLVTVSA",
        "VH_Refanezumab":"EIQLVQSGPELKKPGETNKISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYADDFTGRFAFSLETSASTAYLQISNLKNEDTATYFCARNPINYYGINYEGYVMDYWGQGTLVTVSS",
        "VL_CD28":"DIETLQSPASLAVSLGQRATISCRASESVEYYVTSLMQWYQQKPGQPPKLLIFAASNVESGVPARFSGSGSGTNFSLNIHPVDEDDVAMYFCQQSRKYVPYTFGGGTKLEIK",
        "VL_Campath":"DIKMTQSPSFLSASVGDRVTLNCKASQNIDKYLNWYQQKLGESPKLLIYNTNNLQTGIPSRFSGSGSGTDFTLTISSLQPEDVATYFCLQHISRPRTFGTGTKLELK",
        "VL_Bevacizumab":"DIQMTQTTSSLSASLGDRVIISCSASQDISNYLNWYQQKPDGTVKVLIYFTSSLHSGVPSRFSGSGSGTDYSLTISNLEPEDIATYYCQQYSTVPWTFGGGTKLEIK",
        "VL_Herceptin":"DIVMTQSHKFMSTSVGDRVSITCKASQDVNTAVAWYQQKPGHSPKLLIYSASFRYTGVPDRFTGNRSGTDFTFTISSVQAEDLAVYYCQQHYTTPPTFGGGTKVEIK",
        "VL_Omalizumab":"DIQLTQSPASLAVSLGQRATISCKASQSVDYDGDSYMNWYQQKPGQPPILLIYAASYLGSEIPARFSGSGSGTDFTLNIHPVEEEDAATFYCQQSHEDPYTFGAGTKLEIK",
        "VL_Eculizumab":"DIQMTQSPASLSASVGETVTITCGASENIYGALNWYQRKQGKSPQLLIYGATNLADGMSSRFSGSGSGRQYYLKISSLHPDDVATYYCQNVLNTPLTFGAGTKLELK",
        "VL_Tocilizumab":"DIQMTQTTSSLSASLGDRVTISCRASQDISSYLNWYQQKPDGTIKLLIYYTSRLHSGVPSRFSGSGSGTDYSLTINNLEQEDIATYFCQQGNTLPYTFGGGTKLEIN",
        "VL_Pembrolizumab":"DIVLTQSPASLAVSLGQRAAISCRASKGVSTSGYSYLHWYQQKPGQSPKLLIYLASYLESGVPARFSGSGSGTDFTLNIHPVEEEDAATYYCQHSRDLPLTFGTGTKLELK",
        "VL_Pertuzumab":"DTVMTQSHKIMSTSVGDRVSITCKASQDVSIGVAWYQQRPGQSPKLLIYSASYRYTGVPDRFTGSGSGTDFTFTISSVQAEDLAVYYCQQYYIYPYTFGGGTKLEIK",
        "VL_Ixekizumab":"DVVLTQTPLSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPFTFGSGTKLEIK",
        "VL_Palivizumab":"DIQLTQSPAIMSASPGEKVTMTCSASSSVGYMHWYQQKSSTSPKLWIYDTSKLASGVPGRFSGSGSGNSYSLTISSIQAEDVATYYCFQGSGYPFTFGQGTKLEIK",
        "VL_Certolizumab":"DIVMTQSQKFMSTSVGDRVSVTCKASQNVGTNVAWYQQKPGQSPKALIYSASFLYSGVPYRFTGSGSGTDFTLTISTVQSEDLAEYFCQQYNIYPLTFGAGTKLELK",
        "VL_Idarucizumab":"DVVMTQTPLTLSVTIGQPASISCKSSQSLLYTNGKTYLYWLLQRPGQSPKRLIYLVSKLDSGVPDRFSGSGSGTDFTLKISRVEAEDVGIYYCLQSTHFPHTFGGGTKLEIK",
        "VL_Reslizumab":"DIQMTQSPASLSASLGETISIECLASEGISSYLAWYQQKPGKSPQLLIYGANSLQTGVPSRFSGSGSATQYSLKISSMQPEDEGDYFCQQSYKFPNTFGAGTKLELK",
        "VL_Solanezumab":"DVVMTQTPLSLPVSLGDQASISCRSSQSLIYSDGNAYLHWFLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVETEDLGVYFCSQSTHVPWTFGGGTKLEIK",
        "VL_Lorvotuzumab":"DVLMTQTPLSLPVSLGDQASISCRSSQIIIHSDGNTYLEWFLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLMISRVEAEDLGVYYCFQGSHVPHTFGGGTKLEIK",
        "VL_Pinatuzumab":"DILMTQTPLSLPVSLGDQASISCRSSQSIVHSNGNTFLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSQFPYTFGGGTKVEIK",
        "VL_Etaracizumab":"ELVMTQTPATLSVTPGDSVSLSCRASQSISNHLHWYQQKSHESPRLLIKYASQSISGIPSRFSGSGSGTDFTLSINSVETEDFGMYFCQQSNSWPHTFGGGTKLEIK",
        "VL_Talacotuzumab":"DFVMTQSPSSLTVTAGEKVTMSCKSSQSLLNSGNQKNYLTWYLQKPGQPPKLLIYWASTRESGVPDRFTGSGSGTDFTLTISSVQAEDLAVYYCQNDYSYPYTFGGGTKLEIK",
        "VL_Rovalpituzumab":"SIVMTQTPKFLLVSAGDRVTITCKASQSVSNDVVWYQQKPGQSPKLLIYYASNRYTGVPDRFAGSGYGTDFSFTISTVQAEDLAVYFCQQDYTSPWTFGGGTKLEIR",
        "VL_Clazakizumab":"AYDMTQTPASVSAAVGGTVTIKCQASQSINNELSWYQQKPGQRPKLLIYRASTLASGVSSRFKGSGSGTEFTLTISDLECADAATYYCQQGYSLRNIDNAFGGGTEVVVK",
        "VL_Ligelizumab":"DILLTQSPAILSVSPGERVSFSCRASQSIGTNIHWYQQRTDGSPRLLIKYASESISGIPSRFSGSGSGTEFTLNINSVESEDIADYYCQQSDSWPTTFGGGTKLEIK",
        "VL_Crizanlizumab":"DIVLTQSPASLAVSLGQRATISCKASQSVDYDGHSYMNWYQQKPGQPPKLLIYAASNLESGIPARFSGSGSGTDFTLNIHPVEEEDAATYYCQQSDENPLTFGTGTKLELK",
        "VL_Mogamulizumab":"DVLMTQTPLSLPVSLGDQASISCRSSRNIVHINGDTYLEWYLQRPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSLLPWTFGGGTRLEIR",
        "VL_Refanezumab":"NIMMTQSPSSLAVSAGEKVTMSCKSSHSVLYSSNQKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFTGSGSGTDFTLTIINVHTEDLAVYYCHQYLSSLTFGTGTKLEIK",
        }

    exp_mut = {}
    with open("data/human25_mutations.txt", 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i % 5 == 0:
                exp_mut[lines[i].strip()] = {"VH":lines[i+1].strip(), "VL": lines[i+2].strip(), "VH_cdr": lines[i+3].strip(), "VL_cdr": lines[i+4].strip()}

    # one-shot infer
    vhvl_seq, fixed = {}, {}
    for name, seq in seqs.items():
        fname = name.split("_")[1]
        chain = name.split("_")[0]
        fixed_region = exp_mut[f"{fname}"][f"{chain}_cdr"].split(",")
        if name not in fixed.keys():
            fixed[name] = []
            for i in range(3):
                fixed[name].append(f"{fixed_region[i*2]}-{fixed_region[i*2+1]}")
        if fname not in vhvl_seq.keys():
            vhvl_seq[fname] = {}
        vhvl_seq[fname][chain] = seq
    for name in vhvl_seq.keys():
        top_seq, top_seqs = [], []
        vh, vl = vhvl_seq[name]["VH"], vhvl_seq[name]["VL"]
        fixed_VH, fixed_VL = fixed[f"VH_{name}"], fixed[f"VL_{name}"]
        infer_mut(f"{vh},{vl}", name, fixed_VH=fixed_VH, fixed_VL=fixed_VL, topk=0)

    # iteration infer
    vhvl_seq, fixed = {}, {}
    for name, seq in seqs.items():
        fname = name.split("_")[1]
        chain = name.split("_")[0]
        fixed_region = exp_mut[f"{fname}"][f"{chain}_cdr"].split(",")
        if name not in fixed.keys():
            fixed[name] = []
            for i in range(3):
                fixed[name].append(f"{fixed_region[i*2]}-{fixed_region[i*2+1]}")
        if fname not in vhvl_seq.keys():
            vhvl_seq[fname] = {}
        vhvl_seq[fname][chain] = seq
    j = 0
    for name in vhvl_seq.keys():
        top_seq, top_seqs = [], []
        for n in range(conf.infer_round):
            if n == 0:
                vh, vl = vhvl_seq[name]["VH"], vhvl_seq[name]["VL"]
                fixed_VH, fixed_VL = fixed[f"VH_{name}"], fixed[f"VL_{name}"]
                top_seq = infer_mut(f"{vh},{vl}", name, fixed_VH=fixed_VH, fixed_VL=fixed_VL, topk=conf.top_rank)
                print(n, name, top_seq[0])
                top_seqs = top_seq
                with open(f'design/{name}.txt', 'a') as fin:    
                    fin.write(f"{n},{name},{top_seq[0]}\n")
            if n >= 1:
                for item in top_seqs:
                    vh, vl = item["seq"].split(",")[0], item["seq"].split(",")[1] 
                    fixed_VH, fixed_VL = fixed[f"VH_{name}"], fixed[f"VL_{name}"]
                    top_seq_subset = infer_mut(f"{vh},{vl}", name, fixed_VH=fixed_VH, fixed_VL=fixed_VL, topk=conf.top_rank)
                    for item2 in top_seq_subset:
                        item2['mut'] = item['mut'] + "," + item2["mut"]
                    top_seqs = top_seq_subset             
                top_seqs = sorted(top_seqs, key=lambda x: x['score'], reverse=True)[:conf.top_rank]
                print(n, name, top_seqs[0])
                with open(f'design/{name}.txt', 'a') as fin:    
                    fin.write(f"{n},{name},{top_seqs[0]}\n")
        j += 1