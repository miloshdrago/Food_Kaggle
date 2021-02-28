test_data_dir = '/kaggle/input/food-recognition-challenge/test_set/test_set/'

img_width, img_height = 224, 224

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(128, 128),
    color_mode="rgb",
    shuffle=False,
    class_mode='categorical',
    batch_size=batch_size)


filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(submission_generator)
# We possibly need this: ,steps = np.ceil(nb_samples/batch_size)

# Test if amount of predictions match the number of examples
print(len(predict) == nb_samples)

prediction_names = pd.DataFrame({"img_name": filenames)
prediction_labels = pd.DataFrame(predict)
prediction_df = pd.concat(prediction_names, prediction_labels, axis=1)
display(prediction_df.head())

prediction_df.to_csv("train_labels.csv")
