This algorithm mainly uses image blocking technology and GLCM-EW-TOPSIS to evaluate and sort the weathering degree of historical brick buildings
the hui-chunk.py file processes the image grayscale, equalizes the histogram, and crops the image into sub-images of the same size 
The GLCM.py file extracts image texture features 
topsis.py file for ranking the evaluations 
The final result.py file integrates all the sub-images,  and the topsis.py file is used again to obtain the weathering degree ranking of the sample image
Here are only 3 images for reference,In practical applications, the larger the number of samples, the better the ranking effect
