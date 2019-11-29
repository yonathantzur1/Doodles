import time
import tensorflow as tf
from tensorflow.keras import applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt

image_size = (256, 256)
class_count = 345
epochs = 50
batch_size = 32

lr = 0.001
momentum = 0.9

main_dir = "/home/yonathantzur1/doodle/"
train_dir = "xception/sixth/"

train_data_dir = main_dir + "dataset/train/"
validation_data_dir = main_dir + "dataset/validation/"
test_data_dir = main_dir + "dataset/test/"

model_file = "model.h5"


class LossAccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))


def print_acc(history):
    train_acc = history['accuracy']
    validation_acc = history['val_accuracy']

    # Create the plot
    plt.plot(train_acc)
    plt.plot(validation_acc)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Model Accuracy')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(main_dir + train_dir + "accuracy_history.png")

    # Write result to file
    result = "train accuracy: " + str(train_acc[-1]) + \
             "\nvalidation accuracy: " + str(validation_acc[-1])
    text_file = open(main_dir + train_dir + "train_accuracy.txt", "w")
    text_file.write(str(result))
    text_file.close()


def print_loss(history):
    train_loss = history['loss']
    validation_loss = history['val_loss']

    # Create the plot
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Model Loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(main_dir + train_dir + "loss_history.png")

    # Write result to file
    result = "train loss: " + str(train_loss[-1]) + \
             "\nvalidation loss: " + str(validation_loss[-1])
    text_file = open(main_dir + train_dir + "train_loss.txt", "w")
    text_file.write(str(result))
    text_file.close()


def print_time_log(train_time_in_minutes):
    result = "Training time: " + str(train_time_in_minutes) + " minutes"
    text_file = open(main_dir + train_dir + "train_time.txt", "w")
    text_file.write(str(result))
    text_file.close()


def create_data_augmentations():
    return ImageDataGenerator(rotation_range=5,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              shear_range=0.01,
                              zoom_range=[0.9, 1.1],
                              horizontal_flip=True,
                              vertical_flip=False,
                              fill_mode='reflect',
                              data_format='channels_last')


def create_data():
    return ImageDataGenerator()


def calculate_test_accuracy(model_obj):
    test_gen = create_data()

    test_generator = test_gen.flow_from_directory(test_data_dir,
                                                  target_size=image_size,
                                                  color_mode='rgb',
                                                  batch_size=1,
                                                  class_mode="categorical")

    test_loss, test_acc = model_obj.evaluate_generator(test_generator, verbose=1)
    result = "loss: " + str(test_loss) + "\naccuracy: " + str(test_acc)
    print(result)

    # Write test result to file
    text_file = open(main_dir + train_dir + "test_result.txt", "w")
    text_file.write(str(result))
    text_file.close()


def train_model(model_name=None):
    train_gen = create_data_augmentations()
    validation_gen = create_data()

    train_generator = train_gen.flow_from_directory(train_data_dir,
                                                    target_size=image_size,
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

    validation_generator = validation_gen.flow_from_directory(validation_data_dir,
                                                              target_size=image_size,
                                                              color_mode='rgb',
                                                              class_mode="categorical")

    if model_name:
        model_final = load_model(model_name)
    else:
        base_model = applications.Xception(weights=None,
                                           include_top=False,
                                           input_shape=(image_size[0], image_size[1], 3))

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        fully_connected = Dense(class_count, activation='softmax')(x)
        model_final = Model(inputs=base_model.input, outputs=fully_connected)

    # Compile the final modal with loss and optimizer.
    model_final.compile(loss="categorical_crossentropy",
                        optimizer=optimizers.SGD(lr=lr, momentum=momentum),
                        metrics=["accuracy"])

    model_final.summary()

    step_size_train = train_generator.n // train_generator.batch_size

    # Initializing monitoring params for training.
    history = LossAccHistory()
    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=2, verbose=1, mode='auto')
    network_file_name = main_dir + train_dir + model_file
    checkpoint = ModelCheckpoint(network_file_name, monitor='val_accuracy', verbose=1,
                                 save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

    print("Train...")

    train_start_time = time.time()

    model_history = model_final.fit_generator(generator=train_generator,
                                              steps_per_epoch=step_size_train,
                                              validation_data=validation_generator,
                                              epochs=epochs,
                                              shuffle=True,
                                              callbacks=[history, checkpoint, early])

    train_end_time = time.time()

    train_time_in_minutes = int((train_end_time - train_start_time) / 60)

    print("-------Done-------")

    print_acc(model_history.history)
    print_loss(model_history.history)
    print_time_log(train_time_in_minutes)
    calculate_test_accuracy(load_model(main_dir + train_dir + model_file))


if __name__ == '__main__':
    train_model()
