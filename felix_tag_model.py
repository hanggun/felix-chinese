import os
os.environ['TF_KERAS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from bert4keras.backend import keras, K, gelu_erf
from bert4keras.layers import Dense, Loss, Embedding, SinusoidalPositionEmbedding, concatenate, Dropout
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_gradient_accumulation, extend_with_weight_decay
# from keras import Model
from models import GAU_alpha
import tensorflow as tf

from data_loader import train_loader, dev_loader, max_seqlen, label_map, epochs, batch_size
from evaluate import evaluate


config_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/bert_config.json'
checkpoint_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/bert_model.ckpt'
dict_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/vocab.txt'

bert_model = build_transformer_model(config_path=config_path,
                                     checkpoint_path=checkpoint_path,
                                     model=GAU_alpha,
                                     return_keras_model=False
                                     )
output = Dropout(0.2)(Dense(1024, activation=gelu_erf)(bert_model.output))
tag_logits = Dense(len(label_map))(output)
tag_embedding_fn = Embedding(
                input_dim=len(label_map)+1, # maximum vocab size + 1
                output_dim=128,
                input_length=max_seqlen,
                mask_zero=True)
positional_embedding_fn = SinusoidalPositionEmbedding(output_dim=128)
edit_input = keras.Input(shape=(None,))
point_ids = keras.Input(shape=(None,))
input_mask = keras.Input(shape=(None,))
labels_mask = keras.Input(shape=(None,))
target_ids = keras.Input(shape=(None,))
tag_embedding = tag_embedding_fn(edit_input)
position_embedding = positional_embedding_fn(tag_embedding)
pointer_input = concatenate([output, tag_embedding, position_embedding])
pointer_output = Dense(768, activation=gelu_erf)(pointer_input)
query = Dense(128)(pointer_output)
key = Dense(128)(pointer_output)
pointing_logits = K.batch_dot(query, K.permute_dimensions(key, [0,2,1]))
tag_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
point_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
class FelixTagLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        tag_logits, tag_labels, pointer_logits, pointer_ids, input_mask, labels_mask = inputs
        tag_loss = tag_loss_fn(tag_labels, tag_logits)
        tag_loss = tf.reduce_sum(tag_loss*labels_mask) / tf.reduce_sum(labels_mask)
        self.add_metric(tag_loss, 'tag_loss')
        tag_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
            tag_labels, tag_logits)
        tag_accuracy = tf.reduce_sum(tag_accuracy*labels_mask) / tf.reduce_sum(labels_mask)
        self.add_metric(tag_accuracy, 'tag_accuracy')
        point_loss = point_loss_fn(pointer_ids, pointer_logits)
        point_loss = tf.reduce_sum(point_loss*input_mask) / tf.reduce_sum(input_mask)
        self.add_metric(point_loss, 'point_loss')
        point_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
            pointer_ids, pointer_logits)
        point_accuracy = tf.reduce_sum(point_accuracy * input_mask) / tf.reduce_sum(input_mask)
        self.add_metric(point_accuracy, 'point_accuracy')
        return tag_loss + 0.5*point_loss
model_output = FelixTagLoss([0,2])([tag_logits, edit_input, pointing_logits, point_ids, input_mask, labels_mask])
model = keras.models.Model(bert_model.input+[edit_input, point_ids, input_mask, labels_mask, target_ids], model_output)
AdamW = extend_with_weight_decay(Adam, name='AdamW')
AdamWG = extend_with_gradient_accumulation(AdamW, name='AdamWG')
model.compile(optimizer=AdamW(learning_rate=3e-5, weight_decay_rate=1e-4))
model.summary()

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best_model.weights', save_weights_only=True,
                                                                   monitor='val_tag_accuracy', save_best_only=True)

class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(dev_loader, model)
        if metrics['main'] >= self.best_metric:  # 保存最优
            self.best_metric = metrics['main']
            model.save_weights('saved_model')
        metrics['best'] = self.best_metric
        print(metrics)


evaluator = Evaluator()


if __name__ == '__main__':
    train_steps = 8999//batch_size+1
    dev_steps = 1000//batch_size+1
    # model.load_weights('best_model.weights')
    model.fit(train_loader,
              epochs=epochs,
              steps_per_epoch=train_steps,
              # validation_data=dev_loader,
              # validation_steps=dev_steps,
              callbacks=[evaluator]
              )
else:
    model.load_weights('saved_model')