#!/bin/sh
set -euxo pipefail

# process the raw transition score tensors
python -m score_extraction.process_transition_scores

# merge the extracted scores with the reading measures file
python -m score_extraction.merge_scores