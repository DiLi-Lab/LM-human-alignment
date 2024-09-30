#!/usr/bin/env python3


import pandas as pd
from tqdm import tqdm


def main():
    
    extracted_scores = pd.read_csv('data/extracted_scores.csv', sep='\t')
    rms = pd.read_csv('data/reading_measures_corrected.csv', sep='\t')

    scores_to_keep = [
        'entropies_trunc_wo_nl_wl_joint', 
        'entropies_trunc_wo_nl_wo_punct_wl_joint',
        'surprisal_trunc_wo_nl_wl_sum',
        'surprisal_trunc_wo_nl_wo_punct_wl_sum',
       ]

    scores_dict = dict()
    scores_dict['model'] = list()
    scores_dict['decoding_strategy'] = list()
    scores_dict['item_id'] = list()
    scores_dict['word_id'] = list()
    for score_to_keep in scores_to_keep:
        scores_dict[score_to_keep] = list()   
    
    for idx, row in tqdm(extracted_scores.iterrows()):
        model = row['model']
        item_id = row['item_id']
        decoding_strategy = row['decoding_strategy']

        score_lens = list()

        for score_to_keep_idx, score_to_keep in enumerate(scores_to_keep):
            if score_to_keep in [
                'probabilities_trunc_wo_nl_wl_joint', 
                'probabilities_trunc_wo_nl_wo_punct_wl_joint', 
                'surprisal_trunc_wo_nl_wl_sum',
                'surprisal_trunc_wo_nl_wo_punct_wl_sum',
            ]:
                try:
                    score = eval(row[score_to_keep][7:-1])
                except:
                    breakpoint()
            else:
                score = eval(row[score_to_keep])

            score_lens.append(len(score))
            
            for s_idx, s in enumerate(score):
                if score_to_keep_idx == 0:
                    scores_dict['model'].append(model)
                    scores_dict['decoding_strategy'].append(decoding_strategy)
                    scores_dict['item_id'].append(item_id)
                    scores_dict['word_id'].append(s_idx)
                scores_dict[score_to_keep].append(s)
        
        # assert that all scores have the same length
        assert len(set(score_lens)) == 1
    
    print('finished flattening scores')

    scores_df = pd.DataFrame(scores_dict)

    # lists to hold the new columns:
    entropies_joint, entropies_punct_joint = list(), list()
    surprisal_sum, surprisal_punct_sum = list(), list()

    for idx, row in rms.iterrows():

        print(f'Processing item {idx} of {len(rms)}')

        item_id = row['item_id']
        model = row['model']
        decoding_strategy = row['decoding_strategy']
        word_id = row['word_id']

        entropies_joint.append(scores_df[(scores_df['item_id'] == item_id) & (scores_df['model'] == model) & (scores_df['decoding_strategy'] == decoding_strategy)]['entropies_trunc_wo_nl_wl_joint'].values[word_id])
        entropies_punct_joint.append(scores_df[(scores_df['item_id'] == item_id) & (scores_df['model'] == model) & (scores_df['decoding_strategy'] == decoding_strategy)]['entropies_trunc_wo_nl_wo_punct_wl_joint'].values[word_id])
        surprisal_sum.append(scores_df[(scores_df['item_id'] == item_id) & (scores_df['model'] == model) & (scores_df['decoding_strategy'] == decoding_strategy)]['surprisal_trunc_wo_nl_wl_sum'].values[word_id])
        surprisal_punct_sum.append(scores_df[(scores_df['item_id'] == item_id) & (scores_df['model'] == model) & (scores_df['decoding_strategy'] == decoding_strategy)]['surprisal_trunc_wo_nl_wo_punct_wl_sum'].values[word_id])
    
    rms['entropies_trunc_wo_nl_wl_joint'] = entropies_joint
    rms['entropies_trunc_wo_nl_wo_punct_wl_joint'] = entropies_punct_joint
    rms['surprisal_trunc_wo_nl_wl_sum'] = surprisal_sum
    rms['surprisal_trunc_wo_nl_wo_punct_wl_sum'] = surprisal_punct_sum
    
    print('finished merging scores with rms')

    rms.to_csv('data/reading_measures_corrected_scores.csv', sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
