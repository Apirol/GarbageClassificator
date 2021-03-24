from tensorflow import keras


def preprocessing(image_path, save_path, batch_size: int, augmentation: bool, rotation_range=0):
    IDG = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=augmentation,
                                                       vertical_flip=augmentation, rotation_range=rotation_range)
    return IDG.flow_from_directory(
        image_path, target_size=(192, 256), color_mode='grayscale', classes=None,
        class_mode='binary', batch_size=batch_size, shuffle=False, seed=None,
        save_to_dir=save_path, save_prefix='', save_format='jpeg',
        follow_links=False, subset=None, interpolation='nearest'
    )


def save_images(batches):
    for i in range(0, batches.n):
        batches.next()
