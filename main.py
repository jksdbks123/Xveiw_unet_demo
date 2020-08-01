
import os
from datetime import datetime
from Utils import *
from Data_gen import *
from Unet import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers.schedules import *

class train_model():
    def __init__(self,train_pairs,val_pairs,n_classes = 5,size = (1024,1024),crop_size = None):
        self.size = size
        self.n_classes = n_classes
        self.train_pairs = train_pairs
        self.val_pairs = val_pairs
        self.crop_size = crop_size
        self.epochs = 50
    def train(self):
        train_gen = DataGenerator(
            self.train_pairs,
            size=self.size,
            n_classes=self.n_classes,
            shuffle=True,
            seed=1,
            crop_size=self.crop_size,
            augment=True)

        val_gen = DataGenerator(
            self.val_pairs,
            size = self.size,
            n_classes = self.n_classes,
            shuffle = False,
            crop_size = self.crop_size,
            augment=False
        )
        if self.crop_size is not None:
            model = Unet_model(img_size = self.crop_size , n_classes = self.n_classes)
        else:
            model = Unet_model(img_size = self.size , n_classes = self.n_classes)
        schedule = PiecewiseConstantDecay([2 * len(train_gen)], [1e-5, 1e-6])
        optimizer = RMSprop(learning_rate=schedule)

        # IMPORTANT: make sure the length of this array matches the number of classes
        loss = WeightedCrossEntropy(np.array([0.05, 1.0, 3.0, 3.0, 1.0]))

        metrics = ["acc"]
        for i in range(self.n_classes):
            metrics.append(Precision(class_id=i, name="p_{}".format(i)))
            metrics.append(Recall(class_id=i, name="r_{}".format(i)))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks = [PrintXViewMetrics(n_classes=self.n_classes), TensorBoard(log_dir=log_dir)]
        # if self.checkpoint_dir is not None:
        #     path = os.path.join(self.checkpoint_dir, "checkpoint_{epoch}.h5")
        #     callbacks.append(ModelCheckpoint(path, save_weights_only=True))
        # if self.best is not None:
        #     callbacks.append(
        #         ModelCheckpoint(self.best, monitor="val_loss", save_best_only=True, save_weights_only=True))
        # if self.output_dir is not None:
        #     callbacks.append(SaveOutput(val_gen, self.output_dir))

        model.fit_generator(
            generator=train_gen,
            validation_data=val_gen,
            steps_per_epoch=len(train_gen),
            validation_steps=len(val_gen),
            epochs=self.epochs,
            callbacks=callbacks)

        # if args.save is not None:
        #     model.save_weights(args.save)
if __name__ == '__main__':
    os.chdir(r'/Users/czhui960/Documents/Segdataset')
    dict_pre_post_train1 = json2dict('./train/pairs_dict.json')
    dict_pre_post_train2 = json2dict('./train 2/pairs_dict.json')
    dir_list_train = []
    for key in dict_pre_post_train1.keys():
        dir_list_train += dict_pre_post_train1[key]
    for key in dict_pre_post_train2.keys():
        dir_list_train += dict_pre_post_train2[key]
    dir_list_val = []
    dict_pre_post_val = json2dict('./hold/pairs_dict.json')
    for key in dict_pre_post_val.keys():
        dir_list_val += dict_pre_post_val[key]

    train = train_model(train_pairs=dir_list_train,val_pairs=dir_list_val)
    train.train()