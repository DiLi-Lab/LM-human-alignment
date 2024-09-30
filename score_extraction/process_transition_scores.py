#!/usr/bin/env python3


import os
import os.path
import json

import yaml
import torch
import numpy as np
import pandas as pd

from score_extraction.scores_utils import DecodingScores
from CONSTANTS import path_to_tensors


def main():

    with open('CONSTANTS.yaml', 'r') as f:
        constants = yaml.safe_load(f)
    path_to_tensors = constants['path_to_tensors']
    
    models = ['phi2', 'mistral', 'wizardlm']
    decoding_strategies = ['greedy_search', 'beam_search', 'sampling', 'topk', 'topp']

    path_to_transition_scores = os.path.join(path_to_tensors, 'transition_scores')
    path_to_beam_indices = os.path.join(path_to_tensors, 'beam_indices')
    path_to_sequences = os.path.join(path_to_tensors, 'sequences')
    path_to_model_configs = os.path.join(path_to_tensors, 'model_configs')

    # read in stimuli: needed to get the token idx and word ids
    stimuli = pd.read_csv(os.path.join('data', 'stimuli.csv'), sep='\t')

    # dictionary to hold the scores
    output = dict()
    output['item_id'] = list()
    output['model'] = list()
    output['decoding_strategy'] = list()
    output['entropies'] = list()
    output['entropies_trunc'] = list()
    output['entropies_trunc_wo_nl'] = list()
    output['entropies_trunc_wo_nl_wo_punct'] = list()
    output['entropies_trunc_wo_nl_wl_joint'] = list()
    output['entropies_trunc_wo_nl_wl_first_tok'] = list()
    output['entropies_trunc_wo_nl_wo_punct_wl_joint'] = list()
    output['entropies_trunc_wo_nl_wo_punct_wl_first_tok'] = list()
    output['probabilities'] = list()
    output['probabilities_trunc'] = list()
    output['probabilities_trunc_wo_nl'] = list()
    output['probabilities_trunc_wo_nl_wo_punct'] = list()
    output['probabilities_trunc_wo_nl_wl_joint'] = list()
    output['probabilities_trunc_wo_nl_wo_punct_wl_joint'] = list()
    output['probabilities_trunc_wo_nl_wl_first_tok'] = list()
    output['probabilities_trunc_wo_nl_wo_punct_wl_first_tok'] = list()
    output['norm_scores'] = list()
    output['norm_scores_trunc'] = list()
    output['norm_scores_trunc_wo_nl'] = list()
    output['norm_scores_trunc_wo_nl_wo_punct'] = list()
    output['scores'] = list()
    output['scores_trunc'] = list()
    output['scores_trunc_wo_nl'] = list()
    output['scores_trunc_wo_nl_wo_punct'] = list()
    output['surprisal'] = list()
    output['surprisal_trunc'] = list()
    output['surprisal_trunc_wo_nl'] = list()
    output['surprisal_trunc_wo_nl_wo_punct'] = list()
    output['surprisal_trunc_wo_nl_wl_sum'] = list()
    output['surprisal_trunc_wo_nl_wo_punct_wl_sum'] = list()
    output['surprisal_trunc_wo_nl_wl_first_tok'] = list()
    output['surprisal_trunc_wo_nl_wo_punct_wl_first_tok'] = list()

    # loop over the stimuli
    for idx, row in stimuli.iterrows():

        print(f'Processing item {idx + 1} of {len(stimuli)}')

        item_id = row['item_id']
        model = row['model']
        decoding_strategy = row['decoding_strategy']
        task = row['task']

        config_filename = f'{model}_{decoding_strategy}_model-config.json'
        with open(os.path.join(path_to_model_configs, config_filename), 'r') as f:
            model_config = json.load(f)
        
        Scores = DecodingScores(
            model_config=model_config,
            decoding=decoding_strategy,
        )

        generation_config = eval(row['generation_config'])
        top_k = generation_config['top_k']

        # read in tensors
        base_filename = f'{model}_{decoding_strategy}_{item_id}'
        out_scores = torch.load(os.path.join(path_to_transition_scores, f'{base_filename}_scores.pt'))
        out_sequences = torch.load(os.path.join(path_to_sequences, f'{base_filename}_sequences.pt'))
        
        if decoding_strategy == 'beam_search':
            beam_indices = torch.load(os.path.join(path_to_beam_indices, f'{base_filename}_beam_indices.pt'))
        else:
            beam_indices = None
        
        # compute the subword level scores, probabilities, entropies, etc.
        scores = Scores.scores(
            scores=out_scores,
            sequences=out_sequences,
            top_k=top_k,
            beam_indices=beam_indices,
        )
        norm_scores = Scores.normalized_scores(
            scores=out_scores,
            sequences=out_sequences,
            top_k=top_k,
            beam_indices=beam_indices,
        )
        probabilities = Scores.probabilities(
            scores=out_scores,
            sequences=out_sequences,
            top_k=top_k,
            beam_indices=beam_indices,
        )
        surprisal = Scores.surprisal(
            scores=out_scores,
            sequences=out_sequences,
            top_k=top_k,
            beam_indices=beam_indices,
        )
        entropies = Scores.entropies(
            scores=out_scores,
            sequences=out_sequences,
            top_k=top_k,
            beam_indices=beam_indices,
        )

        ####   POST-PROCESSING   ####

        # truncate the scores so that they match with the truncated sequences (unfinished sentences, eos, newline)
        remove_ctr = row['remove_ctr']
        if remove_ctr == 0:  # generated sequence ended on an end-of-sentence punctuation mark --> remove nothing
            scores_trunc = scores[:, :]
            norm_scores_trunc = norm_scores[:, :]
            probabilities_trunc = probabilities[:, :]
            surprisal_trunc = surprisal[:, :]
            entropies_trunc = entropies[:, :]
        else:  # cut off unfinished part
            scores_trunc = scores[:, :-remove_ctr]
            norm_scores_trunc = norm_scores[:, :-remove_ctr]
            probabilities_trunc = probabilities[:, :-remove_ctr]
            surprisal_trunc = surprisal[:, :-remove_ctr]
            entropies_trunc = entropies[:, :-remove_ctr]
        
        # for poems, the number of newlines was restricted
        cut_nl_idx = row['cut_nl_idx']
        if task == 'poetry':
            scores_trunc_wo_nl = scores_trunc[:, :cut_nl_idx]
            norm_scores_trunc_wo_nl = norm_scores_trunc[:, :cut_nl_idx]
            probabilities_trunc_wo_nl = probabilities_trunc[:, :cut_nl_idx]
            surprisal_trunc_wo_nl = surprisal_trunc[:, :cut_nl_idx]
            entropies_trunc_wo_nl = entropies_trunc[:, :cut_nl_idx]
        
        # subset the scores, probabilities, etc. such that the ones referring to newlines, whitespace, etc are removed
        tok_idx_trunc_wo_nl = eval(row['tok_idx_trunc_wo_nl'])
        scores_trunc_wo_nl = scores_trunc[:, tok_idx_trunc_wo_nl]
        norm_scores_trunc_wo_nl = norm_scores_trunc[:, tok_idx_trunc_wo_nl]
        probabilities_trunc_wo_nl = probabilities_trunc[:, tok_idx_trunc_wo_nl]
        surprisal_trunc_wo_nl = surprisal_trunc[:, tok_idx_trunc_wo_nl]
        entropies_trunc_wo_nl = entropies_trunc[:, tok_idx_trunc_wo_nl]
        
        # scores wtihout those referring to punctuation marks
        tok_idx_trunc_wo_nl_wo_punct = eval(row['tok_idx_trunc_wo_nl_wo_punct'])
        scores_trunc_wo_nl_wo_punct = scores_trunc[:, tok_idx_trunc_wo_nl_wo_punct]
        norm_scores_trunc_wo_nl_wo_punct = norm_scores_trunc[:, tok_idx_trunc_wo_nl_wo_punct]
        probabilities_trunc_wo_nl_wo_punct = probabilities_trunc[:, tok_idx_trunc_wo_nl_wo_punct]
        surprisal_trunc_wo_nl_wo_punct = surprisal_trunc[:, tok_idx_trunc_wo_nl_wo_punct]
        entropies_trunc_wo_nl_wo_punct = entropies_trunc[:, tok_idx_trunc_wo_nl_wo_punct]


        #### POOLING ####

        # pool from subword level to word level (wl = word level)
        word_ids_list_wo_nl = eval(row['word_ids_list_wo_nl'])
        word_ids_list_wo_nl_tensor = torch.LongTensor(word_ids_list_wo_nl)
        word_ids_list_wo_nl_wo_punct = eval(row['word_ids_list_wo_nl_wo_punct'])
        word_ids_list_wo_nl_wo_punct_tensor = torch.LongTensor(word_ids_list_wo_nl_wo_punct)


        # probabilities
        # joint probability: p(w) = p(w1, w2) = p(w1) * p(w2)
        # our variables are not independent, but w1 and w2 already are conditional probabilities so this holds true
        probabilities_trunc_wo_nl_wl_joint = torch.ones(
            len(set(word_ids_list_wo_nl)),
            dtype=probabilities_trunc_wo_nl.dtype,
        ).scatter_reduce_(
            dim=0,
            index=word_ids_list_wo_nl_tensor,
            src=probabilities_trunc_wo_nl.squeeze(),
            reduce='prod',
        )
        probabilities_trunc_wo_nl_wo_punct_wl_joint = torch.ones(
            len(set(word_ids_list_wo_nl_wo_punct)),
            dtype=probabilities_trunc_wo_nl_wo_punct.dtype,
        ).scatter_reduce_(
            dim=0,
            index=word_ids_list_wo_nl_wo_punct_tensor,
            src=probabilities_trunc_wo_nl_wo_punct.squeeze(),
            reduce='prod',
        )
        # for possible analysis, create version where only the probability of the first subword token is used on wl
        probabilities_trunc_wo_nl_wl_first_tok = list()
        for index in range(len(word_ids_list_wo_nl)):
            if index == 0:
                probabilities_trunc_wo_nl_wl_first_tok.append(probabilities_trunc_wo_nl[0][index].item())
            else:
                if word_ids_list_wo_nl[index] == word_ids_list_wo_nl[index - 1]:
                    continue
                else:
                    probabilities_trunc_wo_nl_wl_first_tok.append(probabilities_trunc_wo_nl[0][index].item())
        probabilities_trunc_wo_nl_wo_punct_wl_first_tok = list()
        for index in range(len(word_ids_list_wo_nl_wo_punct)):
            if index == 0:
                probabilities_trunc_wo_nl_wo_punct_wl_first_tok.append(
                    probabilities_trunc_wo_nl_wo_punct[0][index].item())
            else:
                if word_ids_list_wo_nl_wo_punct[index] == word_ids_list_wo_nl_wo_punct[index - 1]:
                    continue
                else:
                    probabilities_trunc_wo_nl_wo_punct_wl_first_tok.append(
                        probabilities_trunc_wo_nl_wo_punct[0][index].item())

        # surprisal
        # pool surprisal to word level: add it up
        # -log(p(a) + p(b)) = -log(a) + -log(b)
        surprisal_trunc_wo_nl_wl_sum = torch.zeros(
            len(set(word_ids_list_wo_nl)),
            dtype=surprisal_trunc_wo_nl.dtype,
        ).scatter_reduce_(
            dim=0,
            index=word_ids_list_wo_nl_tensor,
            src=surprisal_trunc_wo_nl.squeeze(),
            reduce='sum',
        )
        surprisal_trunc_wo_nl_wo_punct_wl_sum = torch.zeros(
            len(set(word_ids_list_wo_nl_wo_punct)),
            dtype=surprisal_trunc_wo_nl_wo_punct.dtype,
        ).scatter_reduce_(
            dim=0,
            index=word_ids_list_wo_nl_wo_punct_tensor,
            src=surprisal_trunc_wo_nl_wo_punct.squeeze(),
            reduce='sum',
        )
        # for possible analysis, create version where only the surprisal of the first subword token is used on wl
        surprisal_trunc_wo_nl_wl_first_tok = list()
        for index in range(len(word_ids_list_wo_nl)):
            if index == 0:
                surprisal_trunc_wo_nl_wl_first_tok.append(surprisal_trunc_wo_nl[0][index].item())
            else:
                if word_ids_list_wo_nl[index] == word_ids_list_wo_nl[index - 1]:
                    continue
                else:
                    surprisal_trunc_wo_nl_wl_first_tok.append(surprisal_trunc_wo_nl[0][index].item())
        surprisal_trunc_wo_nl_wo_punct_wl_first_tok = list()
        for index in range(len(word_ids_list_wo_nl_wo_punct)):
            if index == 0:
                surprisal_trunc_wo_nl_wo_punct_wl_first_tok.append(surprisal_trunc_wo_nl_wo_punct[0][index].item())
            else:
                if word_ids_list_wo_nl_wo_punct[index] == word_ids_list_wo_nl_wo_punct[index - 1]:
                    continue
                else:
                    surprisal_trunc_wo_nl_wo_punct_wl_first_tok.append(surprisal_trunc_wo_nl_wo_punct[0][index].item())

        # entropy
        # for possible analysis, create version where only the entropy of the first subword token is used to on wl
        entropies_trunc_wo_nl_wl_first_tok = list()
        for index in range(len(word_ids_list_wo_nl)):
            if index == 0:
                entropies_trunc_wo_nl_wl_first_tok.append(entropies_trunc_wo_nl[0][index].item())
            else:
                if word_ids_list_wo_nl[index] == word_ids_list_wo_nl[index - 1]:
                    continue
                else:
                    entropies_trunc_wo_nl_wl_first_tok.append(entropies_trunc_wo_nl[0][index].item())
        entropies_trunc_wo_nl_wo_punct_wl_first_tok = list()
        for index in range(len(word_ids_list_wo_nl_wo_punct)):
            if index == 0:
                entropies_trunc_wo_nl_wo_punct_wl_first_tok.append(
                    entropies_trunc_wo_nl_wo_punct[0][index].item())
            else:
                if word_ids_list_wo_nl_wo_punct[index] == word_ids_list_wo_nl_wo_punct[index - 1]:
                    continue
                else:
                    entropies_trunc_wo_nl_wo_punct_wl_first_tok.append(
                        entropies_trunc_wo_nl_wo_punct[0][index].item())
        # joint entropy: right now, it is implemented as adding up individual entropies as proxy for joint entropy
        # out of computational shortcomings
        # in case of beam search, we need to subset the beam indices so it matches the truncated tokens wo newlines
        if decoding_strategy == 'beam_search':
            # remove the newlines and unfinished sentences from the beam indices and sequences
            # (the prompt will still be attached to them at this point)
            input_ids = row['prompt_input_ids']
            if input_ids.startswith('tensor'):
                input_ids = eval(input_ids[7:-1])
            prompt_toks_length = len(input_ids)
            beam_indices_trunc_wo_nl = torch.cat(
                [
                    beam_indices[0][tok_idx_trunc_wo_nl],
                    beam_indices[0][-prompt_toks_length:]
                ]
            ).unsqueeze(0)
            sequences_trunc_wo_nl = torch.cat(
                [
                    out_sequences[0][:prompt_toks_length],
                    out_sequences[0][prompt_toks_length:][tok_idx_trunc_wo_nl]
                ]
            ).unsqueeze(0)
            entropies_trunc_wo_nl_wl_joint = Scores.joint_entropy(
                scores=tuple(out_scores[i] for i in tok_idx_trunc_wo_nl),
                word_ids=word_ids_list_wo_nl,
                top_k=top_k,
                beam_indices=beam_indices_trunc_wo_nl,
                sequences=sequences_trunc_wo_nl,
            )
            beam_indices_trunc_wo_nl_wo_punct = torch.cat(
                [
                    beam_indices[0][tok_idx_trunc_wo_nl_wo_punct],
                    beam_indices[0][-prompt_toks_length:]
                ]
            ).unsqueeze(0)
            sequences_trunc_wo_nl_wo_punct = torch.cat(
                [
                    out_sequences[0][:prompt_toks_length],
                    out_sequences[0][prompt_toks_length:][tok_idx_trunc_wo_nl_wo_punct]
                ]
            ).unsqueeze(0)
            entropies_trunc_wo_nl_wo_punct_wl_joint = Scores.joint_entropy(
                scores=tuple(out_scores[i] for i in tok_idx_trunc_wo_nl_wo_punct),
                word_ids=word_ids_list_wo_nl_wo_punct,
                top_k=top_k,
                beam_indices=beam_indices_trunc_wo_nl_wo_punct,
                sequences=sequences_trunc_wo_nl_wo_punct,
            )
        else:  # all other decoding strategies except beam search
            entropies_trunc_wo_nl_wl_joint = Scores.joint_entropy(
                scores=tuple(out_scores[i] for i in tok_idx_trunc_wo_nl),
                word_ids=word_ids_list_wo_nl,
                top_k=top_k,
            )
            entropies_trunc_wo_nl_wo_punct_wl_joint = Scores.joint_entropy(
                scores=tuple(out_scores[i] for i in tok_idx_trunc_wo_nl_wo_punct),
                word_ids=word_ids_list_wo_nl_wo_punct,
                top_k=top_k,
            )

        # append the scores to the output dictionary
        output['item_id'].append(item_id)
        output['model'].append(model)
        output['decoding_strategy'].append(decoding_strategy)
        output['entropies'].append(entropies)
        output['entropies_trunc'].append(entropies_trunc)
        output['entropies_trunc_wo_nl'].append(entropies_trunc_wo_nl)
        output['entropies_trunc_wo_nl_wo_punct'].append(entropies_trunc_wo_nl_wo_punct)
        output['entropies_trunc_wo_nl_wl_joint'].append(entropies_trunc_wo_nl_wl_joint)
        output['entropies_trunc_wo_nl_wl_first_tok'].append(entropies_trunc_wo_nl_wl_first_tok)
        output['entropies_trunc_wo_nl_wo_punct_wl_joint'].append(entropies_trunc_wo_nl_wo_punct_wl_joint)
        output['entropies_trunc_wo_nl_wo_punct_wl_first_tok'].append(entropies_trunc_wo_nl_wo_punct_wl_first_tok)
        output['probabilities'].append(probabilities)
        output['probabilities_trunc'].append(probabilities_trunc)
        output['probabilities_trunc_wo_nl'].append(probabilities_trunc_wo_nl)
        output['probabilities_trunc_wo_nl_wo_punct'].append(probabilities_trunc_wo_nl_wo_punct)
        output['probabilities_trunc_wo_nl_wl_joint'].append(probabilities_trunc_wo_nl_wl_joint)
        output['probabilities_trunc_wo_nl_wo_punct_wl_joint'].append(probabilities_trunc_wo_nl_wo_punct_wl_joint)
        output['probabilities_trunc_wo_nl_wl_first_tok'].append(probabilities_trunc_wo_nl_wl_first_tok)
        output['probabilities_trunc_wo_nl_wo_punct_wl_first_tok'].append(probabilities_trunc_wo_nl_wo_punct_wl_first_tok)
        output['norm_scores'].append(norm_scores)
        output['norm_scores_trunc'].append(norm_scores_trunc)
        output['norm_scores_trunc_wo_nl'].append(norm_scores_trunc_wo_nl)
        output['norm_scores_trunc_wo_nl_wo_punct'].append(norm_scores_trunc_wo_nl_wo_punct)
        output['scores'].append(scores)
        output['scores_trunc'].append(scores_trunc)
        output['scores_trunc_wo_nl'].append(scores_trunc_wo_nl)
        output['scores_trunc_wo_nl_wo_punct'].append(scores_trunc_wo_nl_wo_punct)
        output['surprisal'].append(surprisal)
        output['surprisal_trunc'].append(surprisal_trunc)
        output['surprisal_trunc_wo_nl'].append(surprisal_trunc_wo_nl)
        output['surprisal_trunc_wo_nl_wo_punct'].append(surprisal_trunc_wo_nl_wo_punct)
        output['surprisal_trunc_wo_nl_wl_sum'].append(surprisal_trunc_wo_nl_wl_sum)
        output['surprisal_trunc_wo_nl_wo_punct_wl_sum'].append(surprisal_trunc_wo_nl_wo_punct_wl_sum)
        output['surprisal_trunc_wo_nl_wl_first_tok'].append(surprisal_trunc_wo_nl_wl_first_tok)
        output['surprisal_trunc_wo_nl_wo_punct_wl_first_tok'].append(surprisal_trunc_wo_nl_wo_punct_wl_first_tok)
    
    output_df = pd.DataFrame(output)
    output_df.to_csv(os.path.join('data', 'extracted_scores.csv'), sep='\t', index=False)


if __name__ == '__main__':
    raise SystemExit(main())
