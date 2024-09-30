#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import string
from score_extraction.EntropyScorer import EntropyScorer
from tqdm import tqdm


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compute entropy for the texts.')
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='The models with which to estimate the entropy.',
    )
    return parser


def process_tsv(
        file_path, 
        scorer, 
        stimuli_path,
        model_name="gpt2", 
        existing_df=None,
        add_prompt: bool = False,
        ):
    df = existing_df if existing_df is not None else pd.read_csv(file_path, sep='\t')

    if add_prompt:
        entropy_col_name = f'entropy_p_{model_name}'
        surprisal_col_name = f'surprisal_{model_name}'
    else:
        entropy_col_name = f'entropy_{model_name}'
        surprisal_col_name = f'surprisal_p_{model_name}'
    
    df[entropy_col_name] = 0.0

    # open the file with the stimuli
    stimuli = pd.read_csv(stimuli_path, sep='\t')


    for _, group in tqdm(df.groupby(['item_id', 'list', 'model', 'decoding_strategy', 'subject_id'])):
        
        words = group['word'].tolist()
        text = ' '.join(words)

        # get the prompt
        model = group['model'].unique().item()
        decoding_strategy = group['decoding_strategy'].unique().item()
        item_id = group['item_id'].unique().item()
        prompt = stimuli.loc[(stimuli['item_id'] == item_id) & (stimuli['model'] == model) & (stimuli['decoding_strategy'] == decoding_strategy)]['prompt'].item()  

        if add_prompt:
            prompt_text = prompt + ' ' + text
            prompt_length = len(prompt.split())
        else:
            prompt_text = text 

        entropies, surprisals, _ = scorer.score(prompt_text, BOS=True)

        if np.isnan(surprisals).any():
            breakpoint()

        assert len(entropies) == len(prompt_text.split())
        assert len(surprisals) == len(prompt_text.split())

        if add_prompt:
            entropies = entropies[prompt_length:]
            surprisals = surprisals[prompt_length:]
        
        assert len(entropies) == len(words), breakpoint()
        assert len(surprisals) == len(words), breakpoint()

        for i, word in enumerate(words):
            df.loc[group.index[i], entropy_col_name] = entropies[i]
            # add surprisal only for wizardlm because it is missing in EMTeC data
            if model_name == 'wizardlm':
                df.loc[group.index[i], surprisal_col_name] = surprisals[i]

    return df


def main():

    args = get_parser().parse_args()
    models = args.models

    input_file = 'data/reading_measures_corrected_scores.csv'
    output_file = 'data/rms_scores_surp_ent.csv'

    for model in models:

        print(f'Processing model {model}')

        scorer = EntropyScorer(model)
        existing_df = pd.read_csv(output_file, sep='\t') if os.path.exists(output_file) else None

        # compute entropy for the text once with the prompt as context for the text
        output_df = process_tsv(
            file_path=input_file,
            scorer=scorer,
            stimuli_path='data/stimuli.csv',
            model_name=model,
            existing_df=existing_df,
            add_prompt=False,
        )

        # compute surprisal 'traditionally', without the prompt as context
        output_df = process_tsv(
            file_path=input_file,
            scorer=scorer,
            stimuli_path='data/stimuli.csv',
            model_name=model,
            existing_df=output_df,
            add_prompt=True,
        )

        output_df.to_csv(output_file, sep='\t', index=False)
        print(f'Output saved to {output_file}')


if __name__ == "__main__":
    raise SystemExit(main())
