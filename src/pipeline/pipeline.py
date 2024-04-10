import cv2
import numpy as np
from transformers import AutoProcessor, VisionEncoderDecoderModel
from PIL import Image


def remove_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove vertical
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (255, 255, 255), 2)
    print('line removing complete')
    return result


def segmentation(image):
    blurred_img = cv2.medianBlur(image, 5)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(blurred_img, kernel, iterations=1)
    output = cv2.dilate(erosion, kernel, iterations=1)
    # load image
    img = output

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold the grayscale image
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 3)
    # use morphology erode to blur horizontally

    # kernel = np.ones((3,3),np.uint8)
    # erosion = cv2.erode(thresh, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # find contours
    cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # Draw contours
    bboxes = []
    for c in cntrs:
        area = cv2.contourArea(c)
        if area > 350:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append((x, y, w, h))

    # crop words
    words = []
    i = 1
    for b in bboxes:
        x, y, w, h = b
        cropped_image = image[y:y + h, x:x + w]
        words.append(cropped_image)
    print('crop complete')
    return words


def text_recognition(word, processor, model):
    print('2')
    image = Image.fromarray(word)
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # recognition
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)
    return generated_text


def process_image(path):
    image = cv2.imread(path)

    nolines = remove_lines(image)
    words_images = segmentation(nolines)

    print('1')
    processor = AutoProcessor.from_pretrained("raxtemur/trocr-base-ru")
    model = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru")

    result = ' '.join(text_recognition(word, processor, model) for word in words_images)

    return result
