from data_loader import constants, _inverse_label_map, tokenizer, insertion_converter, tqdm, dev_loader,\
    max_seqlen, get_mlm_input
import numpy as np
from felix_utils import beam_search
from snippets import compute_metrics, metric_keys
import tensorflow as tf

beam_size = 5


def get_beam_result(prediction, source_token_ids, last_token_index):
    """获得tagging model经过pointer network变换位置的输出，采用beam search的方式"""
    tag_logits, pointing_logits = prediction
    predicted_tags = list(np.argmax(tag_logits, axis=1))
    non_deleted_indexes = set(
        i for i, tag in enumerate(predicted_tags[:last_token_index + 1])
        if _inverse_label_map[int(tag)] not in constants.DELETED_TAGS)
    source_tokens = tokenizer.ids_to_tokens(source_token_ids)
    # get [SEP] indexes
    sep_indexes = set([
        i for i, token in enumerate(source_tokens)
        if token.lower() == constants.SEP.lower() and i in non_deleted_indexes
    ])
    best_sequence = beam_search.beam_search_single_tagging(
        list(pointing_logits), non_deleted_indexes, sep_indexes, beam_size,
        last_token_index, max_seqlen)
    if not best_sequence:
        print('best sequence from beam search is none')
    return best_sequence

def gat_tagging_output(best_sequence, source_token_ids, tags, last_token_index):
    """从beam search中获得的pointing后的路径中获得输出序列和为插入模型设计的输出序列"""
    source_token_ids_set = set(best_sequence)
    out_tokens = []
    out_tokens_with_deletes = []
    for j, index in enumerate(best_sequence):
        token = tokenizer.ids_to_tokens([source_token_ids[index]])
        out_tokens += token
        tag = _inverse_label_map[tags[index]]
        out_tokens_with_deletes += token
        # Add the predicted MASK tokens.
        number_of_masks = insertion_converter.get_number_of_masks(tag)
        # Can not add phrases after last token.
        if j == len(best_sequence) - 1:
            number_of_masks = 0
        masks = [constants.MASK] * number_of_masks
        out_tokens += masks
        out_tokens_with_deletes += masks
        # Find the deleted tokens, which appear after the current token.
        deleted_tokens = []
        for i in range(index + 1, last_token_index + 1):
            if i in source_token_ids_set:
                break
            deleted_tokens.append(source_token_ids[i])
            # Bracket the deleted tokens, between unused0 and unused1.
        if deleted_tokens:
            deleted_tokens = [constants.DELETE_SPAN_START] + list(
                tokenizer.ids_to_tokens(deleted_tokens)) + [
                                 constants.DELETE_SPAN_END
                             ]
            out_tokens_with_deletes += deleted_tokens
    assert (out_tokens_with_deletes[0] == (constants.CLS)), \
        (f' {out_tokens_with_deletes} did not start/end with the correct tokens {constants.CLS}, {constants.SEP}')

    return out_tokens, out_tokens_with_deletes


def evaluate(loader, model):
    total_metrics = {k: 0.0 for k in metric_keys}
    total_len = 0
    for data in tqdm(loader):
        pred = model.predict(data)
        prediction_batch = list(zip(pred[0], pred[1]))
        source_batch = list(zip(*data[0]))
        for source, prediction in zip(source_batch, prediction_batch):
            total_len += 1
            last_token_index = sum(source[4]) - 1 # source[4] is input_mask
            source_token_ids = source[0].numpy()
            tags = source[2].numpy()
            target_ids = source[-1].numpy()
            target_tokens = tokenizer.ids_to_tokens(target_ids[target_ids!=0])

            best_sequence = get_beam_result(prediction, source_token_ids, last_token_index)

            out_tokens, _, = gat_tagging_output(best_sequence, source_token_ids, tags, last_token_index)
            out_tokens = tokenizer.decode(tokenizer.tokens_to_ids(out_tokens))
            target_tokens = tokenizer.decode(tokenizer.tokens_to_ids(target_tokens))
            metrics = compute_metrics(out_tokens, target_tokens)
            for k, v in metrics.items():
                total_metrics[k] += v
    return {k: v / total_len for k, v in total_metrics.items()}


@tf.function(experimental_relax_shapes=True)
def insert_predict(x, insert_model):
     return insert_model(x)

def evaluate_with_tag_and_insert(loader, model, insert_model):
    total_metrics = {k: 0.0 for k in metric_keys}
    total_len = 0
    for data in tqdm(loader):
        pred = model.predict(data)
        prediction_batch = list(zip(pred[0], pred[1]))
        source_batch = list(zip(*data[0]))
        for source, prediction in zip(source_batch, prediction_batch):
            total_len += 1
            last_token_index = sum(source[4]) - 1  # source[4] is input_mask
            source_token_ids = source[0].numpy()
            tags = source[2].numpy()
            target_ids = source[-1].numpy()
            target_tokens = tokenizer.decode(target_ids[target_ids != 0])

            best_sequence = get_beam_result(prediction, source_token_ids, last_token_index)

            out_tokens, out_tokens_with_deletes, = gat_tagging_output(best_sequence, source_token_ids, tags, last_token_index)
            mlm_input = get_mlm_input(out_tokens_with_deletes)
            if mlm_input[3]:
                mlm_input = [np.array(x)[None, :] for x in mlm_input]
                mlm_output = insert_predict(mlm_input, insert_model)[0]
                predicted_tokens = np.argmax(mlm_output, axis=-1)
                current_mask = -1
                new_tokens = []
                in_deletion_bracket = False
                for token in out_tokens_with_deletes:
                    current_mask += 1
                    if token.lower() == constants.DELETE_SPAN_END:
                        in_deletion_bracket = False
                        continue
                    elif in_deletion_bracket:
                        continue
                    elif token.lower() == constants.DELETE_SPAN_START:
                        in_deletion_bracket = True
                        continue

                    if token.lower() == constants.MASK.lower():
                        new_tokens.append(
                            tokenizer.ids_to_tokens([predicted_tokens[current_mask]])[0])
                    else:
                        new_tokens.append(token)
                new_tokens = tokenizer.decode(tokenizer.tokens_to_ids(new_tokens))
                metrics = compute_metrics(list(new_tokens), list(target_tokens))
                for k, v in metrics.items():
                    total_metrics[k] += v
            else:
                new_tokens = tokenizer.decode(tokenizer.tokens_to_ids(out_tokens))
                metrics = compute_metrics(list(new_tokens), list(target_tokens))
                for k, v in metrics.items():
                    total_metrics[k] += v
            # print(f"pred tokens {new_tokens}")
            # print(f"target tokens {target_tokens}")
    return {k: v / total_len for k, v in total_metrics.items()}


if __name__ == '__main__':
    from felix_tag_model import model
    from felix_insert_model import insert_model
    from data_loader import dev_loader_insert
    # print(evaluate(dev_loader, model))
    print(evaluate_with_tag_and_insert(dev_loader, model, insert_model))