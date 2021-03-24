import preprocessing as pr


if __name__ == '__main__':
    image_path_train = "C:/Users/redut/Pictures/Мусорки/Train"
    save_path_train = "C:/Users/redut/PycharmProjects/GarbageClassificator/ProcessedTrain"
    batches_train = pr.preprocessing(image_path_train, save_path_train, 4, True, 30)
    pr.save_images(batches_train)

    image_path_test = "C:/Users/redut/Pictures/Мусорки/Test"
    save_path_test = "C:/Users/redut/PycharmProjects/GarbageClassificator/ProcessedTest"
    batches_test = pr.preprocessing(image_path_test, save_path_test, 1, False, 0)
    pr.save_images(batches_test)
