import os
os.environ['TF_KERAS'] = '1'
import collections
import random
import tensorflow as tf
import felix_utils.felix_constants as constants

from bert4keras.tokenizers import Tokenizer
from felix_utils import utils
from felix_utils import pointing_converter
from felix_utils import insertion_converter
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def bytes_feature(value: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def int64_feature(value: int):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

max_seqlen = 256
max_predictions_per_seq = 40
batch_size = 6
insert_batch_size = 4
epochs = 100
vocab_file = dict_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
label_map = utils.read_label_map('data/csl_10k/label_map.json', use_str_keys=True)
_inverse_label_map = {j:i for i,j in label_map.items()}
converter_tagging = pointing_converter.PointingConverter({}, do_lower_case=True)
converter_insertion = insertion_converter.InsertionConverter(
          max_seq_length=max_seqlen,
          max_predictions_per_seq=max_predictions_per_seq,
          label_map=label_map,
          vocab_file=vocab_file)
print(label_map)
print(_inverse_label_map)
D = []
for line_idx, (sources, target) in enumerate(utils.yield_sources_and_targets('data/csl_10k/train.tsv', 'csl')):
    D.append((sources, target))

train_data, dev_data = train_test_split(D, test_size=0.1, random_state=42, shuffle=True)


def get_mlm_input(out_tokens_with_deletes, target_tokens=None):
    mask_position = []
    mask_target_id = []
    mask_target_weight = []

    for idx, token in enumerate(out_tokens_with_deletes):
        if token != constants.MASK:
            continue
        mask_position.append(idx)
        if target_tokens:
            mask_target_id += tokenizer.tokens_to_ids([target_tokens[idx]])
        else:
            mask_target_id.append(0)
        mask_target_weight.append(1.0)
    # Deleted tokens (bracketed by unused) should have a segment_id of 2.
    unused = False
    segment_ids = []
    for token in out_tokens_with_deletes:
        if token == constants.DELETE_SPAN_START or unused:
            unused = True
            segment_ids.append(1)
        else:
            segment_ids.append(0)
        if token == constants.DELETE_SPAN_END:
            unused = False
    input_mask = [1] * len(out_tokens_with_deletes)
    input_ids = tokenizer.tokens_to_ids(out_tokens_with_deletes)
    assert len(segment_ids) == len(input_ids)
    return input_ids, segment_ids, input_mask, mask_target_id, mask_position, mask_target_weight


def get_tfrecord_data(data, mode='train'):
    writer = tf.io.TFRecordWriter(f'data/csl_10k/{mode}.tfrecord')
    writer_insertion = tf.io.TFRecordWriter(f'data/csl_10k/{mode}_insert.tfrecord')
    total = 0
    for sources, target in tqdm(data):
        # split sequence to tokens
        tokens = tokenizer.tokenize(sources[0].replace(' ', ''), maxlen=max_seqlen)
        output_tokens = tokenizer.tokenize(target.replace(' ', ''), maxlen=max_seqlen)
        # convert tokens to ids
        input_ids = tokenizer.tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        target_ids = tokenizer.tokens_to_ids(output_tokens)
        # compute points by nearest neighbor
        points = converter_tagging.compute_points(tokens, ' '.join(output_tokens))
        if not points:
            print('no points here')
        # 还原原始的point index和label
        cur_idx = points[0].point_index
        org_ids = [0]
        while cur_idx != 0:
            org_ids.append(input_ids[cur_idx])
            if points[cur_idx].added_phrase:
                org_ids.extend(tokenizer.tokens_to_ids(points[cur_idx].added_phrase.split()))
            cur_idx = points[cur_idx].point_index

        # labels是没有可以指向的需要添加的label
        labels = [t.added_phrase for t in points]
        point_indexes = [t.point_index for t in points]
        point_indexes_set = set(point_indexes) #  用于快速比较出现的index
        point_target_tokens = []
        try:
            new_labels = []
            for i, added_phrase in enumerate(labels):
                if i not in point_indexes_set:
                    # 没有指向的位置需要删除
                    new_labels.append(label_map['DELETE'])
                elif not added_phrase:
                    # no added_phrase means not need for insertion
                    new_labels.append(label_map['KEEP'])
                    point_target_tokens.append(input_ids[i])
                else:
                    new_labels.append(label_map['KEEP|' + str(len(added_phrase.split()))])
                    point_target_tokens.append(input_ids[i])
                    for _ in range(len(added_phrase.split())):
                        point_target_tokens.append(tokenizer._token_mask_id)
            labels = new_labels
        except KeyError:
            print(f'added phrase number greater than expected |{added_phrase}')
            continue
        # 这里的label tokens是对应原语句的，labels是label的id
        ######################这部分酌情选择是否使用#########################
        label_tokens = [
            _inverse_label_map.get(label_id, constants.PAD)
            for label_id in labels
        ]
        label_counter = collections.Counter(labels)
        label_weight = {
            label: len(labels) / count / len(label_counter)
            for label, count in label_counter.items()
        }
        # Weight the labels inversely proportional to their frequency.
        labels_mask = [label_weight[label] for label in labels]
        #################################################################


        # sequence mlm source with [MASK] and mlm target with actual word
        # the added [MASK] here used whole word mask strategy
        # Reorder source sentence, add MASK tokens, adds deleted tokens
        # (to both source_tokens and target_tokens).
        masked_tokens, target_tokens = converter_insertion._create_masked_source(
            tokens, labels, point_indexes, output_tokens)
        # for source with no mask, add random mask
        if target_tokens and constants.MASK not in masked_tokens:
            # Don't mask the start or end token.
            indexes = list(range(1, len(masked_tokens) - 1))
            random.shuffle(indexes)
            # Limit MASK to ~15% of the source tokens.
            indexes = indexes[:int(len(masked_tokens) * 0.15)]
            for index in indexes:
                # Do not mask unused tokens
                if masked_tokens[index] != constants.DELETE_SPAN_START and masked_tokens != constants.DELETE_SPAN_END:
                    masked_tokens[index] = constants.MASK
        assert len(target_tokens) == len(masked_tokens)

        mlm_input_ids, mlm_segment_ids, mlm_input_mask,\
        mask_target_id, mask_position, mask_target_weight = get_mlm_input(masked_tokens, target_tokens)

        example = tf.train.Example(features=tf.train.Features(feature={
            'input_ids': int64_list_feature(input_ids),
            'input_mask': int64_list_feature(input_mask),
            'segment_ids': int64_list_feature(segment_ids),
            'labels': int64_list_feature(labels),
            'labels_mask': float_list_feature(labels_mask),
            'point_indexes': int64_list_feature(point_indexes),
            'target_ids': int64_list_feature(target_ids)
        }))
        example_insert = tf.train.Example(features=tf.train.Features(feature={
            'mlm_input_ids': int64_list_feature(mlm_input_ids),
            'mlm_input_mask': int64_list_feature(mlm_input_mask),
            'mlm_segment_ids': int64_list_feature(mlm_segment_ids),
            'mask_target_id': int64_list_feature(mask_target_id),
            'mask_position': int64_list_feature(mask_position),
            'mask_target_weight': float_list_feature(mask_target_weight)
        }))
        writer.write(example.SerializeToString())
        writer_insertion.write(example_insert.SerializeToString())
        total += 1

    print(f'total {total}')
    writer.close()
    writer_insertion.close()


def tfrecord_data_loader(files, batch_size, epochs):
    def map_func(example):
        # feature 的属性解析表
        feature_map = {
            'input_ids': tf.io.VarLenFeature(tf.int64),
            'input_mask': tf.io.VarLenFeature(tf.int64),
            'segment_ids': tf.io.VarLenFeature(tf.int64),
            'labels': tf.io.VarLenFeature(tf.int64),
            'labels_mask': tf.io.VarLenFeature(tf.float32),
            'point_indexes': tf.io.VarLenFeature(tf.int64),
            'target_ids': tf.io.VarLenFeature(tf.int64)
        }
        parsed_example = tf.io.parse_single_example(example, features=feature_map)

        input_ids = tf.cast(tf.sparse.to_dense(parsed_example['input_ids']), tf.int32)
        input_mask = tf.cast(tf.sparse.to_dense(parsed_example['input_mask']), tf.int32)
        segment_ids = tf.cast(tf.sparse.to_dense(parsed_example['segment_ids']), tf.int32)
        labels = tf.cast(tf.sparse.to_dense(parsed_example['labels']), tf.int32)
        labels_mask = tf.sparse.to_dense(parsed_example['labels_mask'])
        point_indexes = tf.cast(tf.sparse.to_dense(parsed_example['point_indexes']), tf.int32)
        target_ids = tf.cast(tf.sparse.to_dense(parsed_example['target_ids']), tf.int32)

        return input_ids, segment_ids, labels, point_indexes, input_mask, labels_mask, target_ids

    def out_process(input_ids, segment_ids, labels, point_indexes, input_mask, labels_mask, target_ids):
        return (input_ids, segment_ids, labels, point_indexes, input_mask, labels_mask, target_ids),

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = (dataset
               .repeat(epochs)
               .map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .padded_batch(batch_size)
               .map(map_func=out_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .prefetch(tf.data.experimental.AUTOTUNE)
               )
    return dataset


def tfrecord_data_loader_insert(files, batch_size, epochs):
    def map_func(example):
        # feature 的属性解析表
        feature_map = {
            'mlm_input_ids': tf.io.VarLenFeature(tf.int64),
            'mlm_input_mask': tf.io.VarLenFeature(tf.int64),
            'mlm_segment_ids': tf.io.VarLenFeature(tf.int64),
            'mask_target_id': tf.io.VarLenFeature(tf.int64),
            'mask_position': tf.io.VarLenFeature(tf.int64),
            'mask_target_weight': tf.io.VarLenFeature(tf.float32)
        }
        parsed_example = tf.io.parse_single_example(example, features=feature_map)

        mlm_input_ids = tf.cast(tf.sparse.to_dense(parsed_example['mlm_input_ids']), tf.int32)
        mlm_input_mask = tf.cast(tf.sparse.to_dense(parsed_example['mlm_input_mask']), tf.int32)
        mlm_segment_ids = tf.cast(tf.sparse.to_dense(parsed_example['mlm_segment_ids']), tf.int32)
        mask_target_id = tf.cast(tf.sparse.to_dense(parsed_example['mask_target_id']), tf.int32)
        mask_position = tf.cast(tf.sparse.to_dense(parsed_example['mask_position']), tf.int32)
        mask_target_weight = tf.sparse.to_dense(parsed_example['mask_target_weight'])

        return mlm_input_ids, mlm_segment_ids, mlm_input_mask, mask_target_id, mask_position, mask_target_weight

    def out_process(mlm_input_ids, mlm_input_mask, mlm_segment_ids, mask_target_id, mask_position, mask_target_weight):
        return (mlm_input_ids, mlm_segment_ids, mlm_input_mask, mask_target_id, mask_position, mask_target_weight),

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = (dataset
               .repeat(epochs)
               .map(map_func=map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .padded_batch(batch_size)
               .map(map_func=out_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .prefetch(tf.data.experimental.AUTOTUNE)
               )
    return dataset


train_loader = tfrecord_data_loader(['data/csl_10k/train.tfrecord'],batch_size=batch_size, epochs=epochs)
dev_loader = tfrecord_data_loader(['data/csl_10k/dev.tfrecord'],batch_size=batch_size, epochs=1)
train_loader_insert = tfrecord_data_loader_insert(['data/csl_10k/train_insert.tfrecord'],batch_size=insert_batch_size, epochs=epochs)
dev_loader_insert = tfrecord_data_loader_insert(['data/csl_10k/dev_insert.tfrecord'],batch_size=insert_batch_size, epochs=1)
if __name__ == '__main__':
    ############ if not generate tfrecord file, please use following scripts
    get_tfrecord_data(train_data, 'train')
    get_tfrecord_data(dev_data, 'dev')


    # for _ in tqdm(train_loader):
    #     pass
    # for _ in tqdm(train_loader_insert):
    #     pass