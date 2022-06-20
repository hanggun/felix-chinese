import os
os.environ['TF_KERAS'] = '1'
from bert4keras.backend import keras, batch_gather
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_gradient_accumulation, extend_with_weight_decay
import tensorflow as tf
from models import GAU_alpha

from data_loader import train_loader_insert, dev_loader, epochs, batch_size
from evaluate import evaluate_with_tag_and_insert


config_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/bert_config.json'
checkpoint_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/bert_model.ckpt'
dict_path = r'D:\PekingInfoResearch\pretrain_models\chinese_GAU-alpha-char_L-24_H-768/vocab.txt'


# with_mlm return softmax result
bert_model = build_transformer_model(config_path=config_path,
                                     checkpoint_path=checkpoint_path,
                                     model=GAU_alpha,
                                     return_keras_model=False,
                                     with_mlm=True)
mlm_logits = bert_model.output
loss_fn = keras.losses.SparseCategoricalCrossentropy(reduction='none')
class MlmLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        mlm_logits, mask_position, mask_target_ids, mask_target_weight = inputs
        mlm_logits = batch_gather(mlm_logits, mask_position)
        mlm_loss = loss_fn(mask_target_ids, mlm_logits)
        mlm_loss = tf.reduce_sum(mlm_loss*mask_target_weight) / tf.reduce_sum(mask_target_weight)
        self.add_metric(mlm_loss, 'mlm_loss')
        mlm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
            mask_target_ids, mlm_logits)
        mlm_accuracy = tf.reduce_sum(mlm_accuracy * mask_target_weight) / tf.reduce_sum(mask_target_weight)
        self.add_metric(mlm_accuracy, 'mlm_accuracy')
        return mlm_loss

mlm_input_mask = keras.Input(shape=(None,))
mask_target_ids = keras.Input(shape=(None,))
mask_position = keras.Input(shape=(None,))
mask_target_weight = keras.Input(shape=(None,))
mlm_output = MlmLoss([0])([mlm_logits, mask_position, mask_target_ids, mask_target_weight])
insert_model = keras.Model(bert_model.input+[mlm_input_mask, mask_target_ids, mask_position, mask_target_weight], mlm_output)
# insert_predict_model = keras.Model(bert_model.input, mlm_output)
AdamW = extend_with_weight_decay(Adam, name='AdamW')
insert_model.compile(optimizer=AdamW(learning_rate=3e-5, weight_decay_rate=1e-4))
insert_model.summary()

class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.best_metric = 0.0

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate_with_tag_and_insert(dev_loader, model, insert_model)
        if metrics['main'] >= self.best_metric:  # 保存最优
            self.best_metric = metrics['main']
            insert_model.save_weights('best_insert_model')
        metrics['best'] = self.best_metric
        print(metrics)


evaluator = Evaluator()


if __name__ == '__main__':
    from felix_tag_model import model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best_insert_model.weights', save_weights_only=True,
                                                                   monitor='val_loss', save_best_only=True)
    train_steps = 8999 // batch_size + 1
    dev_steps = 1000 // batch_size + 1
    insert_model.fit(train_loader_insert,
                     epochs=epochs,
                     steps_per_epoch=train_steps,
                     callbacks=[evaluator])
else:
    insert_model.load_weights('best_insert_model')