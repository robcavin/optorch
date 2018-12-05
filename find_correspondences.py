# coding: utf-8
import cv2 as cv
sift = cv.xfeatures2d.SIFT_create()
imgs = []
for i in range(6) :
    imgs.append(cv.imread('/Users/robcavin/Projects/project/origin_'+str(i)+'.jpg'))
    
    
kps = []
features = []
for img in imgs :
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    (kps_l, features_l) = sift.detectAndCompute(gray,None)
    kps.append(kps_l)
    features.append(features_l)
    
kps_np = []
    
import numpy as np
for kps_l in kps :
    kps_np_l = np.float32([kp.pt for kp in kps_l])
    kps_np.append(kps_np_l)
    
bf = cv.BFMatcher()
matches = bf.knnMatch(features[0],features[1],k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
temp = cv.drawMatchesKnn(imgs[0],kps[0],imgs[1],kps[1],good,None,flags=2)

#import matplotlib.pyplot as plt
#plt.ion()
#plt.imshow(temp)

ptsA = []
ptsB = []
for match in good :
    ptsA.append(kps[0][match[0].queryIdx].pt)
    ptsB.append(kps[1][match[0].trainIdx].pt)

(H, status) = cv.findHomography(np.array(ptsA),np.array(ptsB),cv.RANSAC,10)

result = cv.warpPerspective(imgs[0],H,(imgs[0].shape[1] + imgs[1].shape[1],imgs[0].shape[0]))

import torch
ptsA_truth = torch.tensor(ptsA)
ptsB_truth = torch.tensor(ptsB)

torch.save(ptsA_truth,"ptsA")
torch.save(ptsB_truth,"ptsB")
