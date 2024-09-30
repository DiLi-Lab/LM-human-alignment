

from transformers import AutoModelForCausalLM, GPT2TokenizerFast, GenerationConfig, set_seed, GPT2Config
import torch
import transformers
from typing import Tuple, List, Optional


class DecodingScores:

    """
    Functionalities were partially taken from https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/generation/utils.py#L1067.
    """

    def __init__(
            self,
            model_config,
            decoding: str,  # sampling,

    ):

        self.model_config = model_config
        self.vocab_size = model_config['vocab_size']
        self.decoding = decoding
        if self.decoding not in ['greedy_search', 'sampling', 'topk', 'topp', 'beam_search', 'mbr']:
            raise NotImplementedError

    def entropies(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            top_k: Optional[int] = None,
            beam_indices: Optional[torch.Tensor] = None,
    ):
        if self.decoding in ['greedy_search', 'sampling', 'mbr']:
            flattened = self._flatten_scores(scores=scores)
            unflattened = flattened.reshape(-1, self.vocab_size, flattened.shape[-1])  # [bsz, vocab, steps]
            probabilities = torch.softmax(unflattened, dim=1)
            entropies = -torch.sum(probabilities * torch.log2(probabilities), dim=1)
            return entropies
        elif self.decoding == 'topk':
            return self._entropies_topk(scores=scores, top_k=top_k)
        elif self.decoding == 'topp':
            return self._entropies_topp(scores=scores)
        elif self.decoding == 'beam_search':
            return self._entropies_beam_search(scores=scores, sequences=sequences, beam_indices=beam_indices)

    def scores(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            top_k: Optional[int] = None,
            beam_indices: Optional[torch.Tensor] = None,
    ):
        if self.decoding in ['greedy_search', 'sampling', 'topp', 'beam_search', 'mbr']:
            indices, mask, beam_indices = self._get_indices(scores=scores, sequences=sequences, beam_indices=beam_indices)
            flattened_scores = self._flatten_scores(scores=scores)
            token_scores = flattened_scores.gather(dim=0, index=indices)
            token_scores[mask] = 0
            return token_scores
        elif self.decoding == 'topk':
            assert isinstance(top_k, int)
            token_scores = self._scores_topk(scores=scores, sequences=sequences, top_k=top_k)
            return token_scores

    def normalized_scores(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            top_k: Optional[int] = None,
            beam_indices: Optional[torch.Tensor] = None,
    ):
        if self.decoding in ['greedy_search', 'sampling', 'topp', 'beam_search', 'mbr']:
            indices, mask, beam_indices = self._get_indices(scores=scores, sequences=sequences, beam_indices=beam_indices)
            flattened_norm_scores = self._flattened_normalized_scores(scores=scores)
            token_norm_scores = flattened_norm_scores.gather(dim=0, index=indices)
            token_norm_scores[mask] = 0
            return token_norm_scores
        elif self.decoding == 'topk':
            assert isinstance(top_k, int)
            token_norm_scores = self._scores_topk(scores=scores, sequences=sequences, top_k=top_k, transform='normalized')
            return token_norm_scores

    def probabilities(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            top_k: Optional[int] = None,
            beam_indices: Optional[torch.Tensor] = None,
    ):
        if self.decoding in ['greedy_search', 'sampling', 'topp', 'beam_search', 'mbr']:
            indices, mask, beam_indices = self._get_indices(scores=scores, sequences=sequences, beam_indices=beam_indices)
            flattened_probabilities = self._flattened_probabilities(scores=scores)
            token_probabilities = flattened_probabilities.gather(dim=0, index=indices)
            token_probabilities[mask] = 0
            return token_probabilities
        elif self.decoding == 'topk':
            assert isinstance(top_k, int)
            token_probabilities = self._scores_topk(scores=scores, sequences=sequences, top_k=top_k, transform='probabilities')
            return token_probabilities

    def surprisal(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            top_k: Optional[int] = None,
            beam_indices: Optional[torch.Tensor] = None,
    ):
        probabilities = self.probabilities(scores=scores, sequences=sequences, top_k=top_k, beam_indices=beam_indices)
        return -torch.log2(probabilities)

    def joint_entropy(
            self,
            scores: Tuple[torch.FloatTensor],
            word_ids: List[int],
            top_k: Optional[int] = None,
            beam_indices: Optional[torch.Tensor] = None,
            sequences: Optional[torch.LongTensor] = None,
            threshold: float = 0.000001,
    ) -> List[float]:
        """
            compute the joint entropy of the distributions of subword tokens that belong to the same word.
            for our case, we would have to use the files that end with scores_trunc_wo_nl.pt and use the
            word_ids_list_wo_nl list of word IDs.
        """

        grouped_scores = list()
        if self.decoding in ['greedy_search', 'sampling']:
            grouped_scores = self._group_subwords_to_words(scores=scores, word_ids=word_ids)
        elif self.decoding == 'topk':
            # get the top k scores and their indices in the vocab for each generation step
            flattened = self._flatten_scores(scores=scores)
            unflattened = flattened.reshape(-1, self.vocab_size, flattened.shape[-1])
            topk_values, topk_indices = torch.topk(unflattened, k=top_k, dim=1)
            # get it in the right shape to be grouped into words
            topk_scores = tuple(tensor.squeeze(2) for tensor in torch.split(topk_values, split_size_or_sections=1, dim=2))
            grouped_scores = self._group_subwords_to_words(scores=topk_scores, word_ids=word_ids)
        elif self.decoding == 'topp':
            topp_scores = list()
            for step in scores:
                topp_scores.append(step[step != float('-inf')].unsqueeze(0))
            grouped_scores = self._group_subwords_to_words(scores=tuple(topp_scores), word_ids=word_ids)
        elif self.decoding == 'beam_search':
            indices, mask, beam_indices = self._get_indices(scores=scores, sequences=sequences, beam_indices=beam_indices)
            # scores is a tuple where each tensor is of shape [num beams, vocab]
            scores_correct_beams = list()
            for i in range(beam_indices.shape[-1]):
                scores_correct_beams.append(scores[i][beam_indices[0][i]])
            scores_correct_beams = tuple(scores_correct_beams)
            grouped_scores = self._group_subwords_to_words(scores=scores_correct_beams, word_ids=word_ids)
        word_level_entropies = list()
        for idx, word in enumerate(grouped_scores):
            # create probabilities out of the scores
            if self.model_config['_name_or_path'] == 'mosaicml/mpt-30b-instruct':
                word = [torch.softmax(w.float(), dim=-1).squeeze(0) for w in word]
            else:
                word = [torch.softmax(w, dim=-1).squeeze(0) for w in word]
            # if the current word consists only of one token/subword, do not compute the joint entropy
            if len(word) == 1:
                entropy = -torch.sum(word[0] * torch.log2(word[0]))
                if entropy == -0.0:
                    entropy = torch.tensor(0.)
            else:  # joint entropy between several random variables (subword tokens)
                entropy = self._joint_entropy(word=word, threshold=threshold)
            word_level_entropies.append(entropy.item())

        return word_level_entropies


    def _entropies_topk(self, scores: Tuple[torch.FloatTensor], top_k: int):
        # get the top k scores and their indices in the vocab for each generation step
        flattened = self._flatten_scores(scores=scores)
        unflattened = flattened.reshape(-1, self.vocab_size, flattened.shape[-1])
        topk_values, topk_indices = torch.topk(unflattened, k=top_k, dim=1)
        probabilities = torch.softmax(topk_values, dim=1)
        entropies = -torch.sum(probabilities * torch.log2(probabilities), dim=1)
        return entropies

    def _entropies_topp(self, scores: Tuple[torch.FloatTensor]) -> List[float]:
        # for topp sampling, there is a different no. of possible tokens in each generation step
        # computing the entropy cannot handle the many 0 values when softmaxing over the entire vocab, including the -inf values
        # for each step, we have to extract the tokens among which the probability mass was re-distributed
        # TODO only works for bsz = 1
        entropies = list()
        for step in scores:
            topp_val, topp_idx = self._extract_non_inf_vals_and_idx(scores=step)
            probabilities = torch.softmax(topp_val, dim=-1)
            entropy = -torch.sum(probabilities * torch.log2(probabilities))
            if entropy == -0.0:
                entropy = torch.tensor(0.)
            entropies.append(entropy.item())
        return torch.tensor(entropies).unsqueeze(0)

    def _entropies_beam_search(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            beam_indices: torch.Tensor,
    ) -> List[float]:
        flattened_scores = self._flatten_scores(scores=scores)
        indices, mask, beam_indices = self._get_indices(scores=scores, sequences=sequences, beam_indices=beam_indices)
        probabilities = flattened_scores.reshape(-1, self.model_config['vocab_size'], flattened_scores.shape[-1])
        # probabilities is of shape [num_beams, vocab size, generation steps]
        probabilities = torch.nn.functional.softmax(probabilities, dim=1)
        # entropies is of shape [num_beams, generation steps]
        entropies = -torch.sum(probabilities * torch.log2(probabilities), dim=1)
        # get the entropies of the correct beam
        entropies = entropies.gather(dim=0, index=beam_indices)
        return entropies

    def _scores_topk(self, scores: Tuple[torch.FloatTensor], sequences: torch.LongTensor, top_k: int, transform: str = None):
        flattened_scores = self._flatten_scores(scores=scores, transpose=False)  # shape [steps, bsz *  vocab]
        # get the top k indices and values over the vocab distribution
        topk_val, topk_idx = torch.topk(flattened_scores, k=top_k, dim=1)  # both of shape [steps, top_k]
        if transform == 'normalized':
            topk_val = torch.nn.functional.log_softmax(topk_val, dim=1)
        elif transform == 'probabilities':
            topk_val = torch.softmax(topk_val, dim=1)
        # get the indices / word IDs of the generated sequence, minus the prompt
        steps = flattened_scores.shape[0]
        gen_idx = sequences[:, -steps:]  # bsz * steps
        # get the index positions of the scores in our new distribution (the top_k distribution) that match
        # the generated word ids / generated indices
        idx_in_topk_dist = torch.where(topk_idx == gen_idx.transpose(0, 1))[1]
        # make sure that the generated ids we get from indexing the topk idx is the same as the gen_idx from the
        # model sequences output
        topk_idx_gen_idx = torch.gather(input=topk_idx, dim=1, index=idx_in_topk_dist.unsqueeze(1)).squeeze(1).unsqueeze(0)
        assert torch.equal(gen_idx, topk_idx_gen_idx)
        # get the scores of the generated tokens
        gen_scores = torch.gather(input=topk_val, dim=1, index=idx_in_topk_dist.unsqueeze(1)).squeeze(1).unsqueeze(0)
        return gen_scores


    def _flatten_scores(self, scores: Tuple[torch.FloatTensor], transpose: bool = True):
        # reshape Tuple into [batch size * vocab size, # generation steps], with # generation steps = seq_len - input_len
        if transpose:
            flattened = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)   # [bsz*vocab, steps]
        else:
            flattened = torch.stack(scores).reshape(len(scores), -1)  # [steps, bsz*vocab]
        return flattened

    def _flattened_normalized_scores(self, scores: Tuple[torch.FloatTensor]):
        flattened = self._flatten_scores(scores=scores)  # [bsz, vocab, steps]
        unflattened = flattened.reshape(-1, self.vocab_size, flattened.shape[-1])  # [bsz, vocab, steps]
        unflattened_normalized = torch.nn.functional.log_softmax(unflattened, dim=1)
        flattened_normalized = unflattened_normalized.reshape(-1, unflattened_normalized.shape[-1])  # [bsz*vocab, steps]
        return flattened_normalized

    def _flattened_probabilities(self, scores: Tuple[torch.FloatTensor]):
        flattened = self._flatten_scores(scores=scores)
        unflattened = flattened.reshape(-1, self.vocab_size, flattened.shape[-1])  # [bsz, vocab, steps]
        unflattened_probabilities = torch.softmax(unflattened, dim=1)
        flattened_probabilities = unflattened_probabilities.reshape(-1, unflattened_probabilities.shape[-1])  # [bsz*vocab, steps]
        return flattened_probabilities

    def _get_indices(
            self,
            scores: Tuple[torch.FloatTensor],
            sequences: torch.LongTensor,
            beam_indices: Optional[torch.Tensor] = None,
    ):
        # in case of beam search, we have beam indices; in all other cases, e.g., greedy search, we pretend we have
        # beam search with one beam and simulate beam indices
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1)
            beam_indices = beam_indices.expand(-1, len(scores))

        # in case of beam search, the prompt is signalled as -1 in the beam indices (at the end of each tensor,
        # not at the beginning; early stopped beams are also signalled like this)
        beam_indices_mask = beam_indices < 0

        # max_beam_length is equivalent to the max_new_tokens in e.g. greedy search or sampling
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # set indices of beams that finished early to 0; they will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * self.vocab_size

        # get the input ids / indices of the generated tokens (get rid of prompt indices in sequences)
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices.to(sequences.device)

        return indices, beam_indices_mask, beam_indices

    def _extract_non_inf_vals_and_idx(self, scores: torch.FloatTensor):
        """
        From the output scores of top-p sampling, extract the probability values and their index positions (i.e., input ids)
        :param scores:
        :return:
        """
        # create a boolean mask for non-inf values
        non_inf_mask = (scores != float('-inf'))
        # extract non-inf values using the mask
        non_inf_values = scores[non_inf_mask]
        # extract indices of non-inf values using the mask
        non_inf_indices = torch.arange(scores.size(1)).unsqueeze(0).to(scores.device)[non_inf_mask]
        return non_inf_values, non_inf_indices

    def _group_subwords_to_words(self, scores: Tuple[torch.Tensor], word_ids: List[int]) -> List[List[torch.Tensor]]:
        grouped_tokens = list()
        current_group = list()
        for idx in range(len(word_ids)):
            if idx == 0:
                current_group = [scores[idx]]
            else:
                if word_ids[idx] == word_ids[idx - 1]:
                    current_group.append(scores[idx])
                else:
                    grouped_tokens.append(current_group)
                    current_group = [scores[idx]]
        if current_group:
            grouped_tokens.append(current_group)
        return grouped_tokens

    def _joint_entropy(self, word: List[torch.Tensor], threshold: float, joint_ent_as_sum: bool = True):

        if joint_ent_as_sum:
            # instead of computing the joint entropy properly, add up the individual entropies as proxy of
            # the joint entropy
            entropy = 0.0
            for w in word:
                ent = -torch.sum(w * torch.log2(w))
                entropy += ent
            if entropy == -0.0:
                entropy = torch.tensor(0.)
            return entropy

        else:
            # compute 'proper' joint entropy
            # but in very rare cases, there are 5 or more subwords belonging to the same token
            # even with the treshold this will result in an out of memory error
            # in these cases, just sum up the individual entropies as a proxy
            if self.decoding in ['greedy_search', 'sampling', 'beam_search']:
                # for these three decoding strategies, all scores are considered; for top-k and top-p, the vocab is subset
                # remove all values that fall under a certain probability threshold (memory otherwise will not suffice
                # to compute the joint entropy for more than 2 subword tokens
                word = [w[w > threshold] for w in word]

            # if we have many subwords and there's a token where many vocab positions are retained, do not compute
            # the cartesian product (process will be killed)
            lengths = [w.shape[0] for w in word]
            if len(word) > 2 and any(l > 1000 for l in lengths):
                entropy = 0.0
                for w in word:
                    ent = -torch.sum(w * torch.log2(w))
                    entropy += ent
                return entropy

            # cartesian product of all probabilities of var 1 with all probabilities of var 2 (and var 3, ... for more vars)
            cart_prod = torch.cartesian_prod(*word)
            joint_probs = torch.prod(cart_prod, dim=-1)
            # don't have to account for cases of probability=0 because we filtered them out (either via top-k or top-p,
            # or via the threshold)
            joint_probs_log2 = torch.log2(joint_probs)
            joint_probs_prod = joint_probs * joint_probs_log2
            del joint_probs
            del joint_probs_log2
            entropy = -torch.sum(joint_probs_prod)
            if entropy == -0.0:
                entropy = torch.tensor(0.)
            del joint_probs_prod

        return entropy

