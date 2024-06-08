import os
import sys
sys.path.append('/home/prcstak/Repositories/SEGM-model/')

#!pip install -r SEGM-model/requirements.txt

import cv2
from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering

from huggingface_hub import hf_hub_download

from segm.predictor import SegmPredictor

stages_folder_path = "stages"
words_selected_img = os.path.join(stages_folder_path , "segmentation.jpg")

repo_id = "ai-forever/ReadingPipeline-notebooks"

MODEL_PATH = hf_hub_download(repo_id, "segm/segm_model.ckpt")
CONFIG_PATH = hf_hub_download(repo_id, "segm/segm_config.json")

NUM_THREADS = 8

DEVICE = 'cpu'

RUNTIME = 'Pytorch'

predictor = SegmPredictor(
    model_path=MODEL_PATH,
    config_path=CONFIG_PATH,
    num_threads=NUM_THREADS,
    device=DEVICE,
    runtime=RUNTIME
)

def find_contours(img, to_gray=None):
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed  = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[-2]


def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


def dilate_mask(mask, kernel_size=10):
    kernel  = np.ones((kernel_size,kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


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


def segmentation_model(image):
    pred_data = predictor([image])

    pred_img = pred_data[0]

    show = image.copy()
    cnts = []
    for prediction in pred_img['predictions']:
        if prediction['class_name'] == 'shrinked_text':
            cnts.append(np.array(prediction['polygon'][::-1]))
    contours = sort_contours(cnts, 'top-to-bottom')

    ygs = [box[1]+box[3]/2 for box in contours[1]]


    clusters = list(range(2, 12))
    scores = []
    clusters_centers = []
    for k in clusters:
        # kmeans = KMeans(n_clusters=k).fit(list(zip(ygs)))
        agl = AgglomerativeClustering(n_clusters=k).fit(list(zip(ygs)))
        # preds = kmeans.predict(list(zip(ygs)))
        # clusters_centers.append(kmeans.cluster_centers_)
        scores.append(silhouette_score(list(zip(ygs)), agl.labels_))

    s, k = max(zip(scores, clusters))


    agl = AgglomerativeClustering(n_clusters=k).fit(list(zip(ygs)))
    clusters = agl.labels_
    print("clusters number: " + str(k))

    l = [[] for i in range(k)]
    cls = sorted(set(clusters), key=list(clusters).index)

    for i in range(len(clusters)):
        l[cls.index(clusters[i])].append(contours[0][i])

    l2 = l

    for i in range(len(l)):
        l2[i] = sort_contours(l[i])

    words = []
    for i in l2:
        for contour in i[0]:
            mask = mask_from_contours(image.copy(), [contour])
            expanded = dilate_mask(mask, 15)
            expanded_contour = find_contours(expanded)
            result = image.copy()
            result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = expanded
            x,y,w,h = cv2.boundingRect(expanded_contour[0])
            words.append(result[y:y+h, x:x+w])
            cv2.drawContours(show, expanded_contour, -1, (255, 0, 255), 2)
    
    fig = plt.figure(figsize=(4, 4))
    columns = 1
    rows = len(words)//columns+1
    i = 1
    for w in words:
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(w)
        i+=1
    
    plt.savefig(words_selected_img)
    return words

