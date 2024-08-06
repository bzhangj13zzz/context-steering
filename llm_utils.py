from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch


def get_score_from_generation(generation, get_dist=False):
    scores = generation["scores"][0]
    if get_dist:
        return torch.softmax(scores, dim=-1)
    else:
        return scores


def generate_next_token(
    input: list, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, answer_prepend=[], past_key_values=None
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    model_inputs = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False)
    model_inputs = tokenizer(model_inputs, padding=True, add_special_tokens=False)

    for token in answer_prepend:
        model_inputs["input_ids"].append(token)
        model_inputs["attention_mask"].append(1)

    model_inputs = model_inputs.convert_to_tensors(tensor_type="pt", prepend_batch_axis=True)
    model_inputs = model_inputs.to(device)
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config, past_key_values=past_key_values)

    return generation
