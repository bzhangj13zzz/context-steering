import llm_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


def context_steering(
    prompt_without_context,
    prompt_with_context,
    generated_tokens_so_far,
    lam,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    greedy=True,
    base_cache=None,
    context_cache=None,
):
    base_generation = llm_utils.generate_next_token(
        prompt_without_context,
        model,
        tokenizer,
        answer_prepend=generated_tokens_so_far,
        past_key_values=base_cache,
    )
    
    context_generation = llm_utils.generate_next_token(
        prompt_with_context,
        model,
        tokenizer,
        answer_prepend=generated_tokens_so_far,
        past_key_values=context_cache,
    )

    base_scores = llm_utils.get_score_from_generation(base_generation)
    context_scores = llm_utils.get_score_from_generation(context_generation)
    
    # print(torch.argmax(base_scores))
    # print(torch.argmax(context_scores))

    base_cache = base_generation["past_key_values"]
    context_cache = context_generation["past_key_values"]

    full_scores = (1 + lam) * context_scores - lam * base_scores

    if greedy:
        generated_token = torch.argmax(full_scores)
    else:
        raise NotImplementedError
    
    return generated_token, full_scores, base_cache, context_cache


def steering_generation(prompt_without_context, prompt_with_context, lam, model, tokenizer, max_new_length=20):
    base_cache = None
    context_cache = None
    generated_token = None
    generated_tokens_so_far = []

    while generated_token != tokenizer.eos_token_id and len(generated_tokens_so_far) <= max_new_length:
        generated_token, _, base_cache, context_cache = context_steering(
            prompt_without_context,
            prompt_with_context,
            generated_tokens_so_far,
            lam,
            model,
            tokenizer,
            base_cache=base_cache,
            context_cache=context_cache,
        )
        generated_tokens_so_far.append(generated_token)

    return generated_tokens_so_far


def get_prob_of_generating_sequence_steering(
    prompt_without_context, prompt_with_context, lam, model, tokenizer, sequence
):
    base_cache = None
    context_cache = None

    generation_tokens = tokenizer(sequence, add_special_tokens=False)
    prob_list = []
    logprob_list = []
    for token_idx, token in enumerate(generation_tokens.input_ids):
        # print(token)
        # print(tokenizer.decode(token))
        _, scores, base_cache, context_cache = context_steering(
            prompt_without_context,
            prompt_with_context,
            generation_tokens.input_ids[:token_idx],
            lam,
            model,
            tokenizer,
            base_cache=base_cache,
            context_cache=context_cache,
        )
        scores = scores[0]
        token_prob_dist = torch.softmax(scores, dim=-1)
        token_logprob_dist = torch.log_softmax(scores, dim=-1)
        # prob_list.append(token_prob_dist[token])
        # logprob_list.append(torch.nn.functional.log_softmax(generation["scores"][0], dim=-1)[0])
        # print(token_prob_dist[token])
        # print(token_logprob_dist[token])
        logprob_list.append(token_logprob_dist[token].item())
        prob_list.append(token_prob_dist[token].item())
    return logprob_list, prob_list
