import os
import cv2
import numpy as np
from functools import reduce
import time
from sklearn.cluster import KMeans

def bhattacharyya_ruler(hist1, hist2):
    row = len(hist1)
    col = len(hist1[0])
    row2 = len(hist2)
    if (len(hist2) !=row or len(hist2[0]) != col):
        return False
    # normalization each histogram
    sum = 0
    sum2 = 0

    for i in range(row):
        a1 = reduce(lambda x,y: x+y, hist1[i], 0)
        a2 = reduce(lambda x,y:x+y, hist2[i],0)
        sum += a1
        sum2 += a2
    for i in range(row):
        hist1[i] = list(map(lambda a : float(a)/sum, hist1[i]))
        hist2[i] = list(map(lambda a: float(a) / sum2, hist2[i]))
    # measuring Bhattacharyya distance
    dist = 0
    for i in range(row):
        for j in range(col):
            dist += np.sqrt(hist1[i][j] * hist2[i][j])
    return dist

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def shot_boundary(filepath="src.mp4", framepath="fotolar"):
    """
    save 1 keyframe from evey shot
    :param: video file path
    :return 
    """
    # 创建新文件夹
    if not os.path.exists(framepath):
        os.makedirs(framepath)
    cap = cv2.VideoCapture(filepath) # capture the video from given path
    n_rows = 3  # row number
    n_images_per_row = 3  # to split 9 pieces
    fc = 0  # frame_counter
    images = []  # frames
    hist = []  # histograms
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        hueFrames = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert Hue value
        height, width, ch = hueFrames.shape # find shape of frame and channel
        #cv2.imshow('frame'+str(fc+1), frame)
        roi_height = int(height / n_rows) #to divide 9 pieces find height value for each frame
        roi_width = int(width / n_images_per_row) #to divide 9 pieces find width value for each frame

        #ROI part
        images.append([]) #first row frame_id, column raw piece_id
        hist.append([]) #first row frame_id, column raw piece_id
        for x in range(0, n_rows):
            for y in range(0,n_images_per_row):
                #i am splitting into 9 pieces of each frame and added to the images 2d matrix
                #row defines frame_id and column defines piece_id
                tmp_image=hueFrames[x*roi_height:(x+1)*roi_height, y*roi_width:(y+1)*roi_width]
                images[fc].append(tmp_image)
        # Display the resulting sub-frame and calc local histogram values
        for i in range(0, n_rows*n_images_per_row):
                hist[fc].append(cv2.calcHist([images[fc][i]], [0], None, [256], [0, 256]))
        fc += 1#frame counter 1 up
    dist = [] #distance list

    # calculate bhattacharya dist
    for i in range(0,len(hist)-1):
        dist.append(bhattacharyya_ruler(hist[i],hist[i+1])) # calculate for each 2 frame

    clt = KMeans(n_clusters=2) # n is cluster number
    clt.fit(dist) # calculate Kmeans

    big_center = 1 #select which cluster includes shot frames

    shots = [] #shots list. List will be include frame id

    for i in range(0,len(clt.labels_)):
        if (big_center == clt.labels_[i]):
            #shot frame id is appending to the list
            shots.append(i+1)

    #after we're done we'll destroy everyhting
    cap.release()
    #cv2.destroyAllWindows()

    #from here, this part for store the shot frames under directory
    cap = cv2.VideoCapture(filepath)
    fc = 0 # frame_counter
    i = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        if i < len(shots) and shots[i] == fc:
            # if captured frame is from my shot list then we'll store the frame
            save_path = os.path.join(framepath, "frame_"+str(i).rjust(3, '0')+".png")
            cv2.imwrite(save_path, frame)
            i += 1
        elif i == len(shots): # if we reached out end of the shot list we'll exit
            break
        fc += 1
    cap.release()
    #cv2.destroyAllWindows()
    return fc

if __name__ == "__main__":
    time0 = time.time()
    sbd = shot_boundary("test.mp4")
    print(time.time() - time0)

    
