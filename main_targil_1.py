import cv2
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from processing_class_targil_1 import imageProcessing
from model_configuration_targil_1 import ModelConfiguration
from plotting_results_targil_1 import PlotModelResults
from evaluation_model_targil_1 import EvaluationModel
from sklearn.metrics import accuracy_score
import numpy as np
IVRIT_ALPHABET = [
                    "alef","bet", "gimel", "dalet", "he", "waw","zayin","chet",  "tet","yod",
                    "kaf", "kaf sofit",  "lamed","mem","mem sofit", "nun", "nun sofit", "samech",
                    "ayin", "pe", "pe sofit","tsadi", "tsadi sofit", "qof","resh", "shin", "tav"
                     ]
HDD_FOLDER = './Targil_1/hhd/'
INPUT_SHAPE = (32, 32)
NUM_CLASSES = 27
LAMBDA_A = 0.001
LAMBDA_B = 0.01
DROPOUT_VALUE_1 = 0.05
DROPOUT_VALUE_2 = 0.5
################################################ PRE-PROCESSING #########################################
processor = imageProcessing(IVRIT_ALPHABET)
DICT_ALPHABET_IMAGES = processor.dictionary_letter_images(HDD_FOLDER)
#DICT_ALPHABET_IMAGES = {}
#CREATION OF DICITONNARY : KEY IS LETTER, VALUE IS LIST OF IMAGES FOR THAT LETTER
#AND CONVERSION IMAGES INTO GREYSCALE
# for i, item_letter in enumerate(IVRIT_ALPHABET):
#     folder_path = os.path.join(HDD_FOLDER, str(i))
#     item_letter_images = []
#     if os.path.isdir(folder_path):
#         for filename in os.listdir(folder_path):
#             full_path = os.path.join(folder_path, filename)
#             if os.path.isfile(full_path):
#                 image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
#                 if image is not None:
#                     item_letter_images.append(image)
#                 else:
#                     print("Image is not found")
#     else:
#         print("Folder is not found)")
#     DICT_ALPHABET_IMAGES[item_letter] = item_letter_images
##CHECKING
# total_number_images = 0
# for key, value in DICT_ALPHABET_IMAGES.items():
#     print(f'{key}: {len(value)} images')
#     total_number_images += len(value)
# print(f'Total number of images in HDD base: {total_number_images}')

#ADDING THE PADDING TO IMAGES TO MAKE THEM SQUARED
DICT_ALPHABET_IMAGES_SQUARES = {}
for key, value in DICT_ALPHABET_IMAGES.items():
    item_letter_images_squared = []
    for letter_photo in value:
        letter_photo_square = processor.padding_to_square_images(letter_photo)
        item_letter_images_squared.append(letter_photo_square)
    DICT_ALPHABET_IMAGES_SQUARES[key] = item_letter_images_squared
##CHECKING
# for key, value in DICT_ALPHABET_IMAGES_SQUARES.items():
#     for image in value:
#         print(f"{key} -> {image.shape}, ndim={image.ndim}, type={type(image)}")

# flag = 0
# for key, value in DICT_ALPHABET_IMAGES_SQUARES.items():
#     for image in value:
#         height, width = image.shape
#         if height != width:
#             flag += 1
# print(flag)

#RESIZING THE IMAGES
DICT_ALPHABET_IMAGES_SQUARES_RESIZED = {}
for key, value in DICT_ALPHABET_IMAGES_SQUARES.items():
    list_resized = []
    for letter_photo in value:
        letter_photo_resized = cv2.resize(letter_photo, (32, 32))
        list_resized.append(letter_photo_resized)
    DICT_ALPHABET_IMAGES_SQUARES_RESIZED[key] = list_resized
##CHECKING
# flag = 0
# for key, value in DICT_ALPHABET_IMAGES_SQUARES_RESIZED.items():
#     for image in value:
#         height, width = image.shape
#         if height != 32 and width!=32:
#             flag += 1
# print(flag)

#MAKING THE IMAGES NEGATIVE
DICT_ALPHABET_IMAGES_NEGATIVE = {}
for key, value in DICT_ALPHABET_IMAGES_SQUARES_RESIZED.items():
    list_negative = []
    for letter_photo in value:
        letter_photo_negative = 255 - letter_photo
        list_negative.append(letter_photo_negative)
    DICT_ALPHABET_IMAGES_NEGATIVE[key] = list_negative

# #CHECKING WHETHER CONVERTION INTO NEGATIVE WAS SUCCESSFUL
# #FORWARD INVERSION
# flag = 0
# for value in DICT_ALPHABET_IMAGES_SQUARES_RESIZED.values():
#     for image in value:
#         negative = 255 - image
#         check = np.all((image + negative) == 255)
#         if not check:
#             flag += 1
# print(flag)
#
# #REFERSED INVERSION
# for value in DICT_ALPHABET_IMAGES_NEGATIVE.values():
#     for image in value:
#         original = 255 - image
#         check = np.all((image + original) == 255)
#         if not check:
#             flag += 1
# print(flag)


################################################ PROCESSING #########################################
#DIVISION THE SETS OF IMAGES INTO TRAINING SET (TS), VALIDATION SET (VS), TESTING SET (TestS)
DICT_TRAINING = {}
DICT_VALIDATION = {}
DICT_TEST = {}
for key, value in DICT_ALPHABET_IMAGES_NEGATIVE.items():
    train_set, val_set, test_set = processor.random_division(value)
    DICT_TRAINING[key] = train_set
    DICT_VALIDATION[key] = val_set
    DICT_TEST[key] = test_set
# #CHECKING THE SPLITTING
# print("Number of images in initial dictionary of negative images:")
# for key, value in DICT_ALPHABET_IMAGES_NEGATIVE.items():
#     print(f"{key} -> {len(value)}")
# print("############################################")
# print("Number of negative images for training set:")
# for key, value in DICT_TRAINING.items():
#     print(f"{key} -> {len(value)}")
# print("############################################")
# print("Number of negative images for validation  set:")
# for key, value in DICT_VALIDATION.items():
#     print(f"{key} -> {len(value)}")
# print("############################################")
# print("Number of negative images for testing set:")
# for key, value in DICT_TEST.items():
#     print(f"{key} -> {len(value)}")
# flag = 0
# for key, value in DICT_ALPHABET_IMAGES_NEGATIVE.items():
#     common = len(DICT_TRAINING[key]) + len(DICT_VALIDATION[key]) + len(DICT_TEST[key])
#     if common != len(value):
#         print(f"Mismatch in '{key}': original={len(value)}, split total={common}")
#         flag += 1
# print(f"Here is result of checking: {flag}")
#
# for key, value in DICT_TEST.items():
#     for image in value:
#         print(f"{key} -> {image.shape}, ndim={image.ndim}, type={type(image)}")

############################################### TRAINING ##############################################
# Building  the model.
model = ModelConfiguration(input_shape=INPUT_SHAPE,
                           num_classes=NUM_CLASSES,
                           lambda_a=LAMBDA_A,
                           lambda_b=LAMBDA_B,
                           dropout_value_1=DROPOUT_VALUE_1,
                           dropout_value_2=DROPOUT_VALUE_2)

model_without_regular = model.configuration_base()
model_configuration_L1_lambda_a = model.configuration_L1_lambda_a()
model_configuration_L1_lambda_b = model.configuration_L1_lambda_b()
model_configuration_L2_lambda_a = model.configuration_L2_lambda_a()
model_configuration_L2_lambda_b = model.configuration_L2_lambda_b()
model_configuration_dropout_value_1 = model.configuration_dropout_value_1()
model_configuration_L2_lambda_a_dropout_value_2 = model.configuration_L2_lambda_a_dropout_value_2()
model_configuration_L2_lambda_b_dropout_value_2 = model.configuration_L2_lambda_b_dropout_value_2()


models = [
    model_without_regular,
    model_configuration_L1_lambda_a,
    model_configuration_L1_lambda_b,
    model_configuration_L2_lambda_a,
    model_configuration_L2_lambda_b,
    model_configuration_dropout_value_1,
    model_configuration_L2_lambda_a_dropout_value_2,
    model_configuration_L2_lambda_b_dropout_value_2,
]

for model_item in models:
    model_item.summary()

# #COMPILING MODELS

for model_item in models:
    model_item.compile(
                        optimizer="rmsprop",
                        loss="categorical_crossentropy",
                        metrics=["accuracy"],
                        )
#TRAINING MODELS
#arrange of flattened images for training (x_train)
# and arrange of one-encoded class labels for training (y_train)
x_train_list = []
y_train_list = []
#arrange of images for validation (x_train) and arrange of class labels for validation (y_train)
x_val_list = []
y_val_list = []
#dict comprenabsion where to every letter of alphabet an integer labbel is assigned
label_mapping = {letter: label for label, letter in enumerate(IVRIT_ALPHABET)}

for key, images in DICT_TRAINING.items():
    x_train_list.extend(images)
    y_train_list.extend([label_mapping[key]] * len(images))

for key, images in DICT_VALIDATION.items():
    x_val_list.extend(images)
    y_val_list.extend([label_mapping[key]] * len(images))

#convertion to numpy arrays
x_train = np.array(x_train_list)
y_train = np.array(y_train_list)
x_val = np.array(x_val_list)
y_val = np.array(y_val_list)

# Normalize pixel values (very important)
x_train = x_train / 255.0
x_val = x_val / 255.0

# #flatten images manually
# x_train = x_train.reshape((-1, 32 * 32))
# x_val = x_val.reshape((-1, 32 * 32))

#One-hot encoding of labels
y_train_cat = to_categorical(y_train, num_classes=27)
y_val_cat = to_categorical(y_val, num_classes=27)

training_results = {}

model_names = [
    "without_regular",
    "L1_lambda_a",
    "L1_lambda_b",
    "L2_lambda_a",
    "L2_lambda_b",
    "dropout_value_1",
    "L2_lambda_a_dropout_value_2",
    "L2_lambda_b_dropout_value_2",
]


for model_name, model_instance in zip(model_names, models):
    training_results[model_name] = model_instance.fit(
                        x_train,
                        y_train_cat,
                        epochs=50,
                        batch_size=32,
                        validation_data=(x_val, y_val_cat)
                        )


# training_histories = {
#     "without_regular": TR_without_regular,
#     "L1_lambda_a": TR_L1_lambda_a,
#     "L1_lambda_b": TR_L1_lambda_b,
#     "L2_lambda_a": TR_L2_lambda_a,
#     "L2_lambda_b": TR_L2_lambda_b,
#     "dropout_value_1": TR_dropout_value_1,
#     "L2_lambda_a_dropout_value_2": TR_L2_lambda_a_dropout_value_2,
#     "L2_lambda_b_dropout_value_2": TR_L2_lambda_b_dropout_value_2,
# }



########################## PLOTTING RESULTS AND CHOOSING YHE BEST MODEL ##############
dict_of_object_plots = {}#storage of objects
dict_of_plots = {}#storage of tuples (train_loss, train_acc, valid_loss, valid_acc)
for key, value in training_results.items():
    plotting_model_results = PlotModelResults(value)
    dict_of_object_plots[key] = plotting_model_results
    train_loss, train_acc, valid_loss, valid_acc = plotting_model_results.retrive_TR()
    dict_of_plots[key] = train_loss, train_acc, valid_loss, valid_acc

for key, value in dict_of_object_plots.items():
    value.plot_results([dict_of_plots[key][0], dict_of_plots[key][2]],
                        title=f"{key}: Loss",
                        ylabel="Loss",
                        ylim=[0.0, 75],
                        xlim=[0.0,50],
                        metric_name=["Training Loss", "Validation Loss"],
                        color=["r", "b"],
                        output_dir="./Targil_1/Configurations_plots",
                        save=True)

    value.plot_results([dict_of_plots[key][1], dict_of_plots[key][3]],
                        title=f"{key}: Accuracy",
                        ylabel="Accuracy",
                        ylim=[0.0, 2],
                        xlim=[0.0, 50],
                        metric_name=["Training Accuracy", "Validation Accuracy"],
                        color=["g", "b"],
                        output_dir="./Targil_1/Configurations_plots",
                        save=True)

##################################### EVALUATION OF THE MODEL #################################

evaluation_model = EvaluationModel()
best_model_name = evaluation_model.best_model(dict_of_plots, output_dir=None)


x_test_list = []
y_test_list = []

for key, images in DICT_TEST.items():
    x_test_list.extend(images)
    y_test_list.extend([label_mapping[key]] * len(images))

#convertion to numpy arrays
x_test = np.array(x_test_list)
y_test = np.array(y_test_list)

# Normalization of pixel values (very important)
x_test = x_test / 255.0


#one hot encode labels
y_test_cat = to_categorical(y_test, num_classes=27)

#picking up best model from list "models"
index_of_best_model = model_names.index(best_model_name)
best_model_for_evaluation = models[index_of_best_model]

test_loss, test_accuracy = best_model_for_evaluation.evaluate(x_test, y_test_cat)
print(f"\nTest set accuracy: {test_accuracy:.4f} | Loss: {test_loss:.4f}")

y_predict_probs = best_model_for_evaluation.predict(x_test)
y_predict = np.argmax(y_predict_probs, axis=1)

#Computation of accuracy for each letter
DICT_ACCURACY_PER_LETTER = {}
list_of_accuracies = []

for idx, letter in enumerate(IVRIT_ALPHABET):
    mask = (y_test == idx)
    acc_value = accuracy_score(y_test[mask], y_predict[mask])
    DICT_ACCURACY_PER_LETTER[letter] = acc_value
    list_of_accuracies.append(acc_value)
avg_acc = np.mean(list_of_accuracies)


print("-" * 39)
#print("\n{:^39}".format("Accuracy per letter"))
print("{:^39}".format("Accuracy per letter"))
print("-" * 39)
print("{:<12} | {:>15}".format("Letter", "Accuracy"))
print("-" * 39)
for letter, acc_value in DICT_ACCURACY_PER_LETTER.items():
    print("{:<12} | {:>15.4f}".format(letter, acc_value))
    print("-" * 39)
#print("-" * 39)
print("{:<12} | {:>15.4f}".format("AVG", avg_acc))
print("-" * 39)



#building, plotting and saving in CSV of confusion matrix

evaluation_model.confusion_matrix(y_test, y_predict, IVRIT_ALPHABET, output_dir=None, save=True)


