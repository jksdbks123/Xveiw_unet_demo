import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard


def read_png(filename):
    return tf.image.decode_png(tf.io.read_file(filename))
def write_png(array, filename):
    tf.io.write_file(filename, tf.image.encode_png(array))

def json2dict(filename):  # input: json url
    with open(filename) as f:
        j = json.load(f)
    return j
mask_colors = {
    1: [0, 255, 0],
    2: [255, 255, 0],
    3: [255, 127, 0],
    4: [255, 0, 0]
}
def overlay_mask(base, mask):
    mask_np = mask.numpy()
    output = np.copy(base)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            mask_val = mask_np[i, j]
            if mask_val != 0:
                output[i, j] = mask_colors[mask_val]
    return output


def harmonic_mean(items):
    inv_sum = 0.0
    for item in items:
        inv_sum += (item + 1e-6) ** -1
    return len(items) / inv_sum
class WeightedCrossEntropy:
    # class_weights should be a Numpy array
    def __init__(self, class_weights):
        self.class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.stop_gradient(y_true)
        pixel_losses = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pixel_weights = tf.reduce_mean(self.class_weights * y_true, axis=-1)
        mean_loss = tf.reduce_mean(pixel_weights * pixel_losses)
        return mean_loss

class PrintXViewMetrics(Callback):
    def __init__(self, n_classes=5):
        self.n_classes = n_classes

    def on_epoch_end(self, _, logs):
        f1 = []
        val_f1 = []
        for i in range(self.n_classes):
            p = logs.get("p_{}".format(i))
            r = logs.get("r_{}".format(i))
            val_p = logs.get("val_p_{}".format(i))
            val_r = logs.get("val_r_{}".format(i))

            f1.append(harmonic_mean([p, r]))
            if val_p is not None and val_r is not None:
                val_f1.append(harmonic_mean([val_p, val_r]))

        loc = f1[1]
        print()
        print("loc: {:.4f}".format(loc))
        if self.n_classes > 2:
            damage = harmonic_mean(f1[1:])
            print("damage: {:.4f}".format(damage))
            print("xview2: {:.4f}".format(0.3 * loc + 0.7 * damage))

        if len(val_f1) > 0:
            val_loc = val_f1[1]
            print("val_loc: {:.4f}".format(val_loc))
            if self.n_classes > 2:
                val_damage = harmonic_mean(val_f1[1:])
                print("val_damage: {:.4f}".format(val_damage))
                print("val_xview2: {:.4f}".format(0.3 * val_loc + 0.7 * val_damage))

    class SaveOutput(Callback):
        def __init__(self, gen, output_dir, n_items=50):
            self.gen = gen
            self.output_dir = output_dir
            self.n_items = n_items

        def set_model(self, model):
            self.model = model

        def on_epoch_end(self, epoch, logs):
            for i in range(self.n_items):
                item = self.gen[i]
                pre_post = item[0]

                pre = pre_post[0, :, :, :3]
                pre = tf.cast(pre * 255.0, tf.uint8)
                write_png(pre, os.path.join(self.output_dir, "{}_{}_pre.png".format(epoch, i)))

                post = pre_post[0, :, :, 3:]
                post = tf.cast(post * 255.0, tf.uint8)
                write_png(post, os.path.join(self.output_dir, "{}_{}_post.png".format(epoch, i)))

                true = item[1][0]
                true = tf.argmax(true, axis=-1)
                true = overlay_mask(post, true)
                true = tf.cast(true, tf.uint8)
                write_png(true, os.path.join(self.output_dir, "{}_{}_true.png".format(epoch, i)))

                pred = self.model.predict(pre_post)[0, :, :, :]
                pred = tf.argmax(pred, axis=-1)
                pred = overlay_mask(post, pred)
                pred = tf.cast(pred, tf.uint8)
                write_png(pred, os.path.join(self.output_dir, "{}_{}_pred.png".format(epoch, i)))

    class PrintXViewMetrics(Callback):
        def __init__(self, n_classes=5):
            self.n_classes = n_classes

        def on_epoch_end(self, _, logs):
            f1 = []
            val_f1 = []
            for i in range(self.n_classes):
                p = logs.get("p_{}".format(i))
                r = logs.get("r_{}".format(i))
                val_p = logs.get("val_p_{}".format(i))
                val_r = logs.get("val_r_{}".format(i))

                f1.append(harmonic_mean([p, r]))
                if val_p is not None and val_r is not None:
                    val_f1.append(harmonic_mean([val_p, val_r]))

            loc = f1[1]
            print()
            print("loc: {:.4f}".format(loc))
            if self.n_classes > 2:
                damage = harmonic_mean(f1[1:])
                print("damage: {:.4f}".format(damage))
                print("xview2: {:.4f}".format(0.3 * loc + 0.7 * damage))

            if len(val_f1) > 0:
                val_loc = val_f1[1]
                print("val_loc: {:.4f}".format(val_loc))
                if self.n_classes > 2:
                    val_damage = harmonic_mean(val_f1[1:])
                    print("val_damage: {:.4f}".format(val_damage))
                    print("val_xview2: {:.4f}".format(0.3 * val_loc + 0.7 * val_damage))

    # class TrainValTensorBoard(TensorBoard):
    #     def __init__(self, log_dir='./logs', **kwargs):
    #         train_dir = os.path.join(log_dir, 'training')
    #         super(TrainValTensorBoard, self).__init__(train_dir, **kwargs)
    #         self.val_dir = os.path.join(log_dir, 'validation')
    #
    #     def set_model(self, model):
    #         self.val_writer = tf.summary.create_file_writer(self.val_dir)
    #         super(TrainValTensorBoard, self).set_model(model)
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         logs = logs or {}
    #         validation_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
    #         for name, value in validation_logs.items():
    #             summary = tf.Summary()
    #             summary_value = summary.value.add()
    #             summary_value.simple_value = value.item()
    #             summary_value.tag = name
    #             self.val_writer.add_summary(summary, epoch)
    #         self.val_writer.flush()
    #
    #         logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    #         super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
    #
    #     def on_train_end(self, logs=None):
    #         super(TrainValTensorBoard, self).on_train_end(logs)
    #         self.val_writer.close()