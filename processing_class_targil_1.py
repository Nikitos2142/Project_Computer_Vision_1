import cv2
import random
BORDER_TYPE = cv2.BORDER_CONSTANT
PROPORTIONS = (0.8, 0.1, 0.1)

class imageProcessing:
    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.dictionary = {}

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


