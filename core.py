from typing import List

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from mcg import *
from toolbench import *
from utils import *


class AttackManager:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        adv_tag: str,
        n_samples: int,
        select_tokens_ids: List[int],
        topk: int,
    ):
        """
        Initializes the core components for the adversarial attack.
        model (PreTrainedModel): The pre-trained model to be used.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the pre-trained model.
        adv_tag (str): The tag indicating the content to be optimized for the attack.
        n_samples (int): The number of samples to be taken at each attack step.
        select_tokens_ids (List[int]): The list of token IDs that can be used.
        topk (int): The number of top tokens to be selected at each position.

        Attributes:
            model (PreTrainedModel): The pre-trained model to be used.
            tokenizer (PreTrainedTokenizer): The tokenizer associated with the pre-trained model.
            adv_tag (str): The tag indicating the content to be optimized for the attack.
            n_samples (int): The number of samples to be taken at each attack step.
            select_tokens_ids (List[int]): The list of token IDs that can be used.
            topk (int): The number of top tokens to be selected at each position.
            epoch (int): The current epoch of the optimization process.
            optimization_tokens (Optional[List[int]]): The tokens being optimized, initialized as None.
        """

        self.model = model
        self.tokenizer = tokenizer
        self.adv_tag = adv_tag
        self.n_samples = n_samples
        self.select_tokens_ids = select_tokens_ids
        self.topk = topk
        self.epoch = 0
        self.optimization_tokens = None

    def post_hook(self, new_adv_tag):
        """
        Post hook for the attack manager, used to update the state after each attack step.

        Args:
            new_adv_tag (str): The optimized new tag.
        """
        self.epoch += 1
        self.adv_tag = new_adv_tag
        self.optimization_tokens = None

    def reset(self, adv_tag):
        """
        Resets the attack manager with a new content to be optimized.

        Args:
            adv_tag (str): The new content to be optimized.
        """
        self.epoch = 0
        self.adv_tag = adv_tag
        self.optimization_tokens = None

    def sample_candidates(self, coordinate_grad, space_sep_tokens=True) -> List[str]:
        """
        Samples the candidate replacements for the attack based on the coordinate gradients.

        Args:
            coordinate_grad (Tensor): The coordinate gradients for the optimization tokens.

        Returns:
            List[str]: The list of candidate replacements, sampled based on the coordinate gradients.
        """
        with torch.no_grad():
            sampled_tokens = sample_replacements(
                self.optimization_tokens,
                coordinate_grad,
                self.n_samples * 4,
                self.select_tokens_ids,
                self.topk,
                self.epoch,
            )
        filtered_candidates = filter_candidates(
            self.tokenizer,
            sampled_tokens,
            filter_candidate=True,
            space_sep_tokens=space_sep_tokens,
        )
        filtered_candidates = filtered_candidates[: self.n_samples]

        # Ensure that the original tag is included in the candidates
        filtered_candidates.append(self.adv_tag)
        return filtered_candidates

    @clear_cache_decorator
    def cal_loss(
        self,
        candidates,
        input_ids,
        optimize_slice,
        target=None,
        loss_slice=None,
        batch_size=4,
    ):
        """
        Calculates the loss for the given candidates.

        Args:
            candidates (List[str]): The list of candidate replacements.
            input_ids (Tensor): The input IDs for the model.
            optimize_slice (slice): The slice for the optimization tokens.
            target (Optional[Tensor]): The target tensor for the loss calculation.
            loss_slice (Optional[slice]): The slice for the loss calculation.
            batch_size (int): The batch size for the loss calculation. Defaults to 4.

        Returns:
            Tensor: The loss values for the given candidates
        """
        with torch.no_grad():
            losses = cal_losses(
                self.model,
                self.tokenizer,
                input_ids,
                loss_calculator_wrapper(target if target is None else target),
                optimize_slice,
                candidates,
                batch_size,
            )
        return losses

    @clear_cache_decorator
    def step(self, input_sequence: str, target: torch.Tensor):
        """
        Performs a single step of the attack.

        Args:
            input_sequence (str): The input sequence to be optimized.
            target (Tensor): The target tensor for the loss calculation.

        Returns:
            Tuple[Tensor, Tensor, slice]: The coordinate gradients, input IDs, and the optimize slice.
        """

        str_optimize_slice = slice(
            input_sequence.find(self.adv_tag),
            input_sequence.find(self.adv_tag) + len(self.adv_tag),
        )
        assert str_optimize_slice.start != -1, "未找到adv_tag"

        # Convert the two segments before and after the optimization tag to IDs based on str_optimize_slice
        input_sequence_before = input_sequence[: str_optimize_slice.start]
        input_sequence_after = input_sequence[str_optimize_slice.stop :]
        optimize_sequence = input_sequence[str_optimize_slice]
        input_sequence_ids_before = self.tokenizer(
            input_sequence_before, add_special_tokens=False
        ).input_ids
        input_sequence_ids_after = self.tokenizer(
            input_sequence_after, add_special_tokens=False
        ).input_ids

        optimize_sequence_ids = self.tokenizer(
            optimize_sequence, add_special_tokens=False
        ).input_ids
        optimize_slice = slice(
            len(input_sequence_ids_before) + 1,
            len(input_sequence_ids_before) + 1 + len(optimize_sequence_ids),
        )
        input_ids = (
            [self.tokenizer.cls_token_id]
            + input_sequence_ids_before
            + optimize_sequence_ids
            + input_sequence_ids_after
            + [self.tokenizer.sep_token_id]
        )

        assert len(input_ids) == len(
            self.tokenizer(input_sequence, add_special_tokens=True).input_ids
        ), "Sequence length mismatch"

        input_ids = torch.tensor(input_ids).to(self.model.device)

        if self.optimization_tokens is None:
            self.optimization_tokens = input_ids[optimize_slice]

        coordinate_grad = token_gradients(
            self.model,
            input_ids,
            optimize_slice,
            loss_calculator_wrapper(target),
        )
        return coordinate_grad, input_ids, optimize_slice


class AttackCoreRetriever:
    def __init__(
        self,
        adv_tag_retriever: str,
        fake_api,
        queries: List[str],
        encoder_attacker: AttackManager,
    ):
        """
        Initializes the core components for the retriever attack.

        Args:
            adv_tag_retriever (str): The tag to be optimized for the attack.
            fake_api: The fake API to be optimized.
            queries (List[str]): The list of queries to be used for the attack.
            encoder_attacker (AttackManager): The attack manager for the encoder.
        """

        self.fake_api = fake_api
        self.adv_tag_retriever = adv_tag_retriever
        self.queries = queries
        self.encoder_attacker = encoder_attacker

    def set_target(self, target: torch.Tensor):
        self.target = target.to(self.encoder_attacker.model.device)

    def get_doc(self):
        """
        Get the document format of the fake API.
        """
        return convert_tool_json_to_corpus(self.fake_api)

    def update_fake_api(self, old_tag, new_tag):
        self.fake_api["api_description"] = self.fake_api["api_description"].replace(
            old_tag, new_tag
        )

    def choose_candidate(self, candidates, loss, greater_is_better=False):
        if greater_is_better:
            best_id = loss.argmax().item()
        else:
            best_id = loss.argmin().item()
        new_candidate = candidates[best_id]
        return new_candidate, best_id

    def step(self, epoch):
        doc = self.get_doc()

        retriever_coordinate_grad, retriever_input_ids, retriever_optimize_slice = (
            self.encoder_attacker.step(doc, self.target)
        )
        candidates = self.encoder_attacker.sample_candidates(retriever_coordinate_grad)
        retriever_loss = self.encoder_attacker.cal_loss(
            candidates,
            retriever_input_ids,
            retriever_optimize_slice,
            target=self.target,
            batch_size=128,
        )
        candidate, best_id = self.choose_candidate(candidates, retriever_loss)
        self.update_fake_api(self.adv_tag_retriever, candidate)
        self.encoder_attacker.post_hook(candidate)
        self.adv_tag_retriever = candidate
        print(f"Epoch{epoch}:")
        print(
            f"len(candidates): {len(candidates)}\tloss: {retriever_loss[best_id].item()}\tadv_tag: {candidate}"
        )
