import cv2
import random
import os
BORDER_TYPE = cv2.BORDER_CONSTANT
PROPORTIONS = (0.8, 0.1, 0.1)

class imageProcessing:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.dictionary = {}

    # CREATION OF DICITONNARY : KEY IS LETTER, VALUE IS LIST OF IMAGES FOR THAT LETTER
    # AND CONVERSION IMAGES INTO GREYSCALE
    def dictionary_letter_images(self, hdd_base):
        DICT_ALPHABET_IMAGES = {}
        for i, item_letter in enumerate(self.alphabet):
            folder_path = os.path.join(hdd_base, str(i))
            item_letter_images = []
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    full_path = os.path.join(folder_path, filename)
                    if os.path.isfile(full_path):
                        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            item_letter_images.append(image)
                        else:
                            print("Image is not found")
            else:
                print("Folder is not found)")
            DICT_ALPHABET_IMAGES[item_letter] = item_letter_images
        return DICT_ALPHABET_IMAGES



    def padding_to_square_images(self, image):
        height, width = image.shape
        if height == width:
            return image
        if height > width:
            padding_value = (height - width) // 2
            left = padding_value
            right = (height - width) - padding_value
            top = bottom = 0
        elif height < width:
            padding_value = (width - height) // 2
            top = padding_value
            bottom = (width - height) - padding_value
            left = right = 0
        squared_image = cv2.copyMakeBorder(
            src=image,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=BORDER_TYPE,
        )
        return squared_image



    def random_division(self, list_images, proportions = PROPORTIONS):
        data_for_shuffle = list_images.copy()
        random.shuffle(data_for_shuffle)
        number_of_images = len(data_for_shuffle)
        #calculation of start and end of the splitted sets
        train_set_end = int(number_of_images * proportions[0])
        validation_set_end = train_set_end + int(number_of_images * proportions[1])
        #splitting
        training_set = data_for_shuffle[:train_set_end]
        validation_set = data_for_shuffle[train_set_end:validation_set_end]
        testing_set = data_for_shuffle[validation_set_end:]
        return training_set, validation_set, testing_set


