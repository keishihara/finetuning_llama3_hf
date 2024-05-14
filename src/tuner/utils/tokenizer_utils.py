from transformers import AutoModel, AutoTokenizer

from tuner import DEFAULT_PAD_TOKEN
from tuner.utils.distributed_utils import print_on_rank_0


def smart_tokenizer_and_embedding_resize(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    *,
    smart_assign: bool = True,
    verbose: bool = True,
) -> None:
    """Resize tokenizer and embedding."""

    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN

    # Llama 3 Instruct models only
    if getattr(model, 'name_or_path', None) \
       in ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-70B-Instruct']:
        tokenizer.eos_token = '<|eot_id|>'

    # if pad_token is the same as some other special token, replace it
    if tokenizer.pad_token is not None:
        special_tokens = tokenizer.special_tokens_map.copy()
        special_tokens.pop('pad_token', None)
        if tokenizer.pad_token in list(special_tokens.values()):
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN

    if not special_tokens_dict:
        return

    if verbose:
        print_on_rank_0(f'Adding missing special tokens or replacing pad_token: {special_tokens_dict}')
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    if num_new_tokens:
        # resize embeddings
        model.resize_token_embeddings(len(tokenizer))

        if smart_assign:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
