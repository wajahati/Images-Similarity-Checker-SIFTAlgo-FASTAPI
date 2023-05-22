# -*- coding: utf-8 -*-
"""
Created on Sun May 14 02:11:08 2023

@author: mr.laptop
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from API import Similarity


import cv2
import numpy as np
import gc
import requests
from io import BytesIO
from PIL import Image

app = FastAPI()

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def compare_images(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.10*n.distance :
          if m.distance==0:
            good_matches.append(1)
    return len(good_matches)

def are_images_similar(url_list1, url_list2):
    threshold = 100
    similar_images = []
    for url1 in url_list1:
        img1 = download_image(url1)
        for url2 in url_list2:
            img2 = download_image(url2)
            num_good_matches = compare_images(img1, img2)
            if num_good_matches > threshold:
                similar_images.append((url1, url2, num_good_matches))
                return similar_images
            del img2
            gc.collect()
        del img1
        gc.collect()
    return similar_images


@app.post('/similarityCheck')
def similarity_check(data: Similarity):
    inp_imgs = [cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_COLOR) for img in data.inpImg]
    pro_urls = data.proImg

    result = are_images_similar(inp_imgs, pro_urls)
    output_dict = {"similarity": len(result) > 0, "similar_images": result}
    return JSONResponse(content=output_dict)


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
