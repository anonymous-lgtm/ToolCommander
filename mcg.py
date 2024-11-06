from typing import List

import numpy as np
import torch

from utils import *


def token_gradients(
    model,
    input_ids,
    optimize_slice,
    cal_loss,
):
    """
    Calculate the gradients of the loss with respect to the tokens in the optimization slice.

    Args:
        model (torch.nn.Module): The model to attack.
        input_ids (torch.Tensor): The input ids.
        optimize_slice (slice): The slice of the input_ids to optimize.
        cal_loss (Callable): The loss function.

    Returns:
        torch.Tensor: The gradients of the loss with respect to the tokens in the optimization slice.
    """
    embed_weights = get_embedding_layer(model)

    one_hot = torch.zeros(
        input_ids[optimize_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[optimize_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()

    # Re-calculating the embeddings for the input_ids, enables the gradients to be calculated
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, : optimize_slice.start, :],
            input_embeds,
            embeds[:, optimize_slice.stop :, :],
        ],
        dim=1,
    )

    del embed_weights, input_embeds, embeds
    last_hidden_state = model(inputs_embeds=full_embeds).last_hidden_state
    pooling_result = mean_pooling(last_hidden_state, model.device)
    loss = cal_loss(pooling_result)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    grad = grad.detach().cpu()

    return grad


@clear_cache_decorator
def sample_replacements(
    optimization_tokens,
    coordinate_grad,
    batch_size,
    select_token_ids,
    topk,
    epoch,
    c_min=4,
):
    """
    Use MCG to sample the replacement tokens.

    Args:
        optimization_tokens (torch.Tensor): The optimization tokens.
        coordinate_grad (torch.Tensor): The coordinate gradients.
        batch_size (int): The batch size.
        select_token_ids (List[int]): The list of token ids to select.
        topk (int): The number of topk tokens to sample.
        epoch (int): current epoch
        c_min (int): minimum c value
    """
    c = int(max(coordinate_grad.shape[0] / (1.2 * (epoch + 1)), c_min))

    # Remove the disabled tokens in the coordinate_grad
    mask = torch.zeros(coordinate_grad.shape[1], dtype=torch.bool)
    mask[select_token_ids] = True
    coordinate_grad[:, ~mask] = np.infty

    # Sample the topk tokens for each position
    top_indices = (-coordinate_grad).topk(topk, dim=1).indices
    optimization_tokens = optimization_tokens.to(coordinate_grad.device)
    original_control_tokens = optimization_tokens.repeat(batch_size, 1)
    # Sample the c positions with the highest sum(TopK)
    new_token_pos = torch.randint(
        0, optimization_tokens.shape[0], (batch_size, c), device=coordinate_grad.device
    )
    new_token_topk_pos = torch.randint(
        0, topk, (batch_size, c, 1), device=coordinate_grad.device
    )
    new_token_val = torch.gather(top_indices[new_token_pos], 2, new_token_topk_pos)
    new_optimization_tokens = original_control_tokens.scatter_(
        1, new_token_pos, new_token_val.squeeze(-1)
    )
    return new_optimization_tokens


@clear_cache_decorator
def cal_losses(
    model,
    tokenizer,
    input_ids,
    cal_loss,
    optimize_slice: slice,
    candidates: List[str],
    batch_size: int,
):
    """
    Calculate the losses for the candidates.

    Args:
        model (torch.nn.Module): The model to attack.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        input_ids (torch.Tensor): The input ids.
        cal_loss (Callable): The loss function.
        optimize_slice (slice): The slice of the input_ids to optimize.
        candidates (List[str]): The list of candidates.
        batch_size (int): The batch size.
    """

    def forward(model, input_ids, attention_mask):
        """
        Forward pass through the model.
        """
        losses = []
        for i in range(0, input_ids.shape[0], batch_size):
            batch_input_ids = input_ids[i : i + batch_size]
            if attention_mask is not None:
                batch_attention_mask = attention_mask[i : i + batch_size]
            else:
                batch_attention_mask = None
            losses.append(
                cal_loss(
                    mean_pooling(
                        model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                        ).last_hidden_state,
                        model.device,
                    )
                )
            )
            del batch_input_ids, batch_attention_mask
            import gc

            gc.collect()
        return torch.concat(losses, dim=-1)

    test_ids = []
    max_len = 0
    for candidate in candidates:
        candidate_ids = torch.tensor(
            tokenizer(candidate, add_special_tokens=False).input_ids,
            device=model.device,
        )
        max_len = max(candidate_ids.shape[0], max_len)
        test_ids.append(candidate_ids)
    nested_ids = torch.nested.nested_tensor(test_ids)
    test_ids = torch.nested.to_padded_tensor(
        nested_ids, tokenizer.pad_token_id, output_size=(len(test_ids), max_len)
    ).to(model.device)
    locs = (
        torch.arange(optimize_slice.start, optimize_slice.stop)
        .repeat(test_ids.shape[0], 1)
        .to(model.device)
    )
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids,
    )
    attn_mask = (ids != tokenizer.pad_token_id).type(ids.dtype)
    del locs, test_ids, nested_ids
    losses = forward(model=model, input_ids=ids, attention_mask=attn_mask)
    return losses


@clear_cache_decorator
def filter_candidates(
    tokenizer, candidates, filter_candidate=True, space_sep_tokens=True
):
    """
    Filter the candidates.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        candidates (List[str]): The list of candidates.
        filter_candidate (bool): Whether to filter the candidates.
        space_sep_tokens (bool): Whether the tokens are space separated.
    """
    filtered_candidates = set()
    for i in range(candidates.shape[0]):
        if space_sep_tokens:
            elements = []
            for j in candidates[i]:
                elements.append(tokenizer.decode(j, skip_special_tokens=True).strip())
            decoded_str = " ".join(elements)
        else:
            decoded_str = tokenizer.decode(candidates[i], skip_special_tokens=True)
        decoded_str = " " + decoded_str.strip()
        if filter_candidate:
            if (
                len(tokenizer(decoded_str, add_special_tokens=False).input_ids)
                < len(candidates[i]) * 1.1
            ):
                filtered_candidates.add(decoded_str)
        else:
            filtered_candidates.add(decoded_str)
    return list(filtered_candidates)
