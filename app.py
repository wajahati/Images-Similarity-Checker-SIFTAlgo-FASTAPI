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

def download_image(url, max_size=800):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.thumbnail((max_size, max_size), Image.ANTIALIAS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def compare_images(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches)

def are_images_similar(url_list1, url_list2):
    threshold = 50
    similar_images = []
    for url1 in url_list1:
        img1 = download_image(url1)
        for url2 in url_list2:
            img2 = download_image(url2)
            num_good_matches = compare_images(img1, img2)
            if num_good_matches > threshold:
                similar_images.append((url1, url2))
            img2 = None
            gc.collect()
        img1 = None
        gc.collect()
    return similar_images

@app.post('/similarityCheck')
def profanityCheck(data:Similarity):
    inpImgs = data.inpImg
    proImgs = data.proImg
    
    result = are_images_similar(inpImgs, proImgs)
    output_dict = {"similarity": len(result) > 0, "similar_images": result}
    return JSONResponse(content=output_dict)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
