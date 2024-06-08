import cv2
import os
import numpy as np
from transformers import AutoProcessor, VisionEncoderDecoderModel
from matplotlib import pyplot as plt
from PIL import Image
from autocorrect import Speller

from segmentation_model import segmentation_model

stages_folder_path = "stages"
hwt_section_img = os.path.join(stages_folder_path , "selection.jpg")
nolines_img = os.path.join(stages_folder_path , "nolines.jpg")
words_selected_img = os.path.join(stages_folder_path , "segmentation.jpg")


def rotate(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    [vx,vy,x,y] = cv2.fitLine(cntsSorted[1], cv2.DIST_L2,0,0.01,0.01)
    x_axis      = np.array([1, 0])    # unit vector in the same direction as the x axis
    your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
    dot_product = np.dot(x_axis, your_line)
    angle_2_x   = float(np.arccos(dot_product))
    degree = 180*angle_2_x/3.14
    rotated = img.copy()
    
    if abs(degree)>1.6:
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # rotate our image by 45 degrees around the center of the image
        M = cv2.getRotationMatrix2D((cX, cY), -round(180*angle_2_x/3.14), 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

    return rotated


def crop_recomendation_section(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    min_section_width = image.shape[1]*0.7
    cntsFiltred = [contours[x] for x in range(len(contours)) if hierarchy[0][x][3] == 0 and cv2.boundingRect(contours[x])[2] > min_section_width]
    cntsSorted = sorted(cntsFiltred, key=lambda x: cv2.boundingRect(x)[3], reverse=True)
    cntsSorted = sorted(cntsSorted[1:7], key=lambda x: cv2.boundingRect(x)[1], reverse=True)

    mask = np.zeros(image.shape[0:2], np.uint8)
    mask = cv2.drawContours(mask, cntsSorted, 0, (255,255,255), cv2.FILLED)
    mask_inv = 255 - mask
    bckgnd = np.full_like(image, (255,255,255))
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    bckgnd_masked = cv2.bitwise_and(bckgnd, bckgnd, mask=mask_inv)
    result = cv2.add(image_masked, bckgnd_masked)

    x, y, w, h = cv2.boundingRect(cntsSorted[0])
    cropped_image = result[y:y + h, x:x + w]

    cv2.imwrite(hwt_section_img, cropped_image)
    
    return cropped_image


def remove_lines(image):
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15,1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    result = cv2.add(temp2, image)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, vertical_kernel)
    result = cv2.add(temp2, result)

    cv2.imwrite(nolines_img, result)

    return result


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def segmentation2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    binarized_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    
    hpp = np.sum(binarized_image, axis=1)

    threshold = (np.mean(hpp)+((np.max(hpp)-np.min(hpp))/2))/2

    peaks = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])

    peaks_indexes = np.array(peaks)[:, 0].astype(int)

    diff_between_consec_numbers = np.diff(peaks_indexes) # difference between consecutive numbers
    indexes_with_larger_diff = np.where(diff_between_consec_numbers > 1)[0].flatten()
    peak_groups = np.split(peaks_indexes, indexes_with_larger_diff)
    # remove very small regions, these are basically errors in algorithm because of our threshold value
    peak_groups = [item for item in peak_groups if len(item) > 10]

    seperated_images = []
    for index, sub_image_index in enumerate(peak_groups):
        sub_image = image[sub_image_index[0]-12:sub_image_index[1]+12]
        seperated_images.append(sub_image)

    words = []

    for i in range(len(seperated_images)-1):
        line = seperated_images[i+1]
        binary = cv2.threshold(cv2.cvtColor(line, cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        # find the vertical projection by adding up the values of all pixels along rows
        vertical_projection = np.sum(binary, axis=0)

        height =  vertical_projection.max()#line.shape[0]
        ## we will go through the vertical projections and
        ## find the sequence of consecutive white spaces in the image
        whitespace_lengths = []
        whitespace = 0
        for vp in vertical_projection:
            if vp == height:
                whitespace = whitespace + 1
            elif vp != height:
                if whitespace != 0:
                    whitespace_lengths.append(whitespace)
                whitespace = 0 # reset whitepsace counter.

        avg_white_space_length = (np.max(whitespace_lengths)-np.min(whitespace_lengths))//2

        ## find index of whitespaces which are actually long spaces using the avg_white_space_length
        whitespace_length = 0
        divider_indexes = []
        for index, vp in enumerate(vertical_projection):
            if vp == height:
                whitespace_length = whitespace_length + 1
            elif vp != height:
                if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                    divider_indexes.append(index-int(whitespace_length/2))
                    whitespace_length = 0 # reset it

        # lets create the block of words from divider_indexes
        divider_indexes = np.array(divider_indexes)
        dividers = np.column_stack((divider_indexes[:-1],divider_indexes[1:]))

        for index, window in enumerate(dividers):
            word = line[:,window[0]:window[1]]
            words.append(word)

    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = len(words)//columns+1
    i = 1
    for w in words:
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(w)
        i+=1
    
    plt.savefig(words_selected_img)

    return words


def segmentation4(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    binarized_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    
    hpp = np.sum(binarized_image, axis=1)

    threshold = (np.mean(hpp)+((np.max(hpp)-np.min(hpp))/2))/2

    peaks = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])

    peaks_indexes = np.array(peaks)[:, 0].astype(int)

    diff_between_consec_numbers = np.diff(peaks_indexes)
    indexes_with_larger_diff = np.where(diff_between_consec_numbers > 1)[0].flatten()
    peak_groups = np.split(peaks_indexes, indexes_with_larger_diff)
    peak_groups = [item for item in peak_groups if len(item) > 10]

    seperated_images = []
    for index, sub_image_index in enumerate(peak_groups):
        sub_image = image[sub_image_index[0]-12:sub_image_index[1]+12]
        seperated_images.append(sub_image)

    words = []

    for i in range(len(seperated_images)-1):
        line = seperated_images[i+1]
        gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 10))
        morph = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel)

        vertical_projection = np.sum(morph, axis=0)

        height =  0
        whitespace_lengths = []
        whitespace = 0
        for vp in vertical_projection:
            if vp == height:
                whitespace = whitespace + 1
            elif vp != height:
                if whitespace != 0:
                    whitespace_lengths.append(whitespace)
                whitespace = 0 # reset whitepsace counter.
        whitespace_lengths.append(whitespace)

        
        whitespace_length = 0
        divider_indexes = []
        for index, vp in enumerate(vertical_projection):
            if vp == height:
                whitespace_length = whitespace_length + 1
            elif vp != height:
                if whitespace_length != 0:
                    divider_indexes.append(index-int(whitespace_length/2))
                    whitespace_length = 0 # reset it
        if whitespace_length != 0  and whitespace_length > 10:
            divider_indexes.append(index-int(whitespace_length/2))

        
        divider_indexes = np.array(divider_indexes)
        dividers = np.column_stack((divider_indexes[:-1],divider_indexes[1:]))

        for index, window in enumerate(dividers):
            word = line[:,window[0]:window[1]]
            words.append(word)

    fig = plt.figure(figsize=(8, 8))
    columns = 5
    rows = len(words)//columns+1
    i = 1
    for w in words:
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(w)
        i+=1
    
    plt.savefig(words_selected_img)

    return words


def segmentation3(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(image,(5,5),0)
    binarized_image = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    
    hpp = np.sum(binarized_image, axis=1)

    threshold = (np.mean(hpp)+((np.max(hpp)-np.min(hpp))/2))/2

    peaks = []
    for i, hppv in enumerate(hpp):
        if hppv < threshold:
            peaks.append([i, hppv])

    peaks_indexes = np.array(peaks)[:, 0].astype(int)

    diff_between_consec_numbers = np.diff(peaks_indexes) # difference between consecutive numbers
    indexes_with_larger_diff = np.where(diff_between_consec_numbers > 1)[0].flatten()
    peak_groups = np.split(peaks_indexes, indexes_with_larger_diff)
    # remove very small regions, these are basically errors in algorithm because of our threshold value
    peak_groups = [item for item in peak_groups if len(item) > 10]

    seperated_images = []
    for index, sub_image_index in enumerate(peak_groups):
        sub_image = image[sub_image_index[0]-12:sub_image_index[1]+12]
        seperated_images.append(sub_image)

    words = []

    for i in range(len(seperated_images)-1):
        line = seperated_images[i+1]
        blurred_img = cv2.medianBlur(line, 5)

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
        (srted, bboxes) = sort_contours(cntrs)

        # crop words
        line_words = []
        i = 1
        for b in bboxes:
            x, y, w, h = b
            cropped_image = line[y:y + h, x:x + w]
            line_words.append(cropped_image)
        
        words+=line_words

    return words

def segmentation1(image):
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

    result = words[::-1] 
    return result


def text_recognition(word, processor, model):
    image = Image.fromarray(word).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # recognition
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def process_image(path):
    image = cv2.imread(path)

    rotated = rotate(image)
    cropped = crop_recomendation_section(rotated)
    nolines = remove_lines(cropped)
    # words_images = segmentation4(image)

    words_images = segmentation_model(nolines)

    processor = AutoProcessor.from_pretrained("raxtemur/trocr-base-ru")
    model = VisionEncoderDecoderModel.from_pretrained("raxtemur/trocr-base-ru")

    result = ' '.join(text_recognition(word, processor, model) for word in words_images)
    print(result)

    spell = Speller('ru')

    return spell(result)
