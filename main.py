import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from scipy import optimize
import matplotlib.pyplot as plt


def make_aug_data(image_path, save_path, batch_size):
    IDG = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, rotation_range=40,
                                                       vertical_flip=True, rescale=1. / 255)
    batches = IDG.flow_from_directory(
        image_path, target_size=(192, 256), color_mode='grayscale', classes=None,
        class_mode='binary', batch_size=batch_size, shuffle=True, seed=None,
        save_to_dir=save_path, save_prefix='', save_format='jpeg',
        follow_links=False, subset=None, interpolation='nearest'
    )
    save_images(batches)


def save_images(batches):
    for i in range(10):
        batches.next()



if __name__ == '__main__':
    """
    image_path_train_full = "C:/Users/redut/Pictures/Мусорки_2/Полные"
    save_path_train_full = "C:/Users/redut/PycharmProjects/GarbageClassificator/Dataset/Full/"

    image_path_train_not_full = "C:/Users/redut/Pictures/Мусорки_2/Неполные"
    save_path_train_not_full = "C:/Users/redut/PycharmProjects/GarbageClassificator/Dataset/NotFull/"

    make_aug_data(image_path_train_full, save_path_train_full, 21)
    make_aug_data(image_path_train_not_full, save_path_train_not_full, 21)
    
    """

    image_path = 'C:/Users/redut/Desktop/Dataset'
    train_ds = keras.preprocessing.image_dataset_from_directory(
        image_path,
        shuffle=True,
        validation_split=0.2,
        color_mode='grayscale',
        subset="training",
        seed=123,
        image_size=(192, 256),
        batch_size=16)

    val_ds = keras.preprocessing.image_dataset_from_directory(
        image_path,
        shuffle=True,
        validation_split=0.2,
        color_mode='grayscale',
        subset="validation",
        seed=123,
        image_size=(192, 256),
        batch_size=16)

    image_path_test = 'C:/Users/redut/PycharmProjects/GarbageClassificator/Dataset/'
    test_ds = keras.preprocessing.image_dataset_from_directory(
        image_path_test,
        color_mode='grayscale',
        seed=123,
        image_size=(192, 256),
        batch_size=16)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    train_ds = train_ds
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 2
    model = keras.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(32, 7, padding='same', activation='relu', input_shape=(192, 256, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 7, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 7, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

    #model.summary()

    epochs = 15
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    #model.evaluate(test_ds)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()



