import cv2
import numpy as np
import os
# Open the image files.
cnt=0
st2 = r'C:\Users\prana\OneDrive\Desktop\img regstr'
for j in range(10,100):
  st3 = 'IMG_02'+str(j)+'_'+str(5)+'.tif'
  img5_path = os.path.join(st2, 'old_data', st3)
  img2_color = cv2.imread(img5_path)

  for p in range(1,5):
    s = 'IMG_02'+str(j)+'_'+str(p)+'.tif'
    img_path = os.path.join(st2, 'old_data', s)
    img1_color = cv2.imread(img_path)
  # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)
    # print(type(matches))  
    matches=list(matches)
    # Sort matches on the basis of their Hamming distance.
    matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
      p1[i, :] = kp1[matches[i].queryIdx].pt
      p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                    homography, (width, height))

    # Save the output.
    gray_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)

# Get the total number of pixels
    total_pixels = gray_img.size

# Get the pixel values
    pixels = gray_img.flatten()

# Initialize a counter for pure black pixels
    black_pixels = 0

# Loop through each pixel and check if it's pure black
    for pixel in pixels:
        if pixel == 0:
            black_pixels += 1
    black_percentage = (black_pixels / total_pixels) * 100
    if black_percentage>4:
       cnt+=1
       transformed_img=img2
    if len(transformed_img.shape)==3:
      transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
    height, width = transformed_img.shape

# Set the desired border size
    border_size = 20

# Create a black border around the image
    for i in range(border_size):
          transformed_img[i, :] = 0  # set pixel value to 10
          transformed_img[height-i-1, :] =0  # set pixel value to 10
          transformed_img[:, i] = 0  # set pixel value to 10
          transformed_img[:, width-i-1] =0  # set pixel value to 10

# Calculate the percentage of pure black pixels
    o="intermediate/output_"
    o+=str(j)+"_"
    o+=str(p)
    o+=".jpg"
    cv2.imwrite(o, transformed_img)

  img5=img2
  # print( img5.shape)
  height, width = img5.shape

# Set the desired border size
  border_size = 20

# Create a black border around the image
  for i in range(border_size):
    img5[i, :] = 0   # Top border
    img5[height-1-i, :] = 0  # Bottom border
    img5[:, i] = 0   # Left border
    img5[:, width-1-i] = 0  # Right border

  cv2.imwrite("intermediate/output_"+str(j)+"_5.jpg",img5)
  import cv2
  import numpy as np
  n_bands = 5
  st3='output_'+str(j)+'_1'+'.jpg'
  img4_path= os.path.join(st2, 'intermediate', st3)
  img= cv2.imread(img4_path, cv2.IMREAD_GRAYSCALE)
  # img= MB_img[:,:,0]
  img_shape=img.shape[:2]
  MB_img = np.zeros((img_shape[0],img_shape[1],n_bands))  # 3 dimensional dummy array with zeros
  MB_img[:,:,0]=img
  for i in range(1,5):
      #print(i)
      st3= 'output_'+str(j)+'_'+str(i+1)+'.jpg'
      img_path6= os.path.join(st2, 'intermediate', st3)
      # img_path6=st2+'intermediate/output_'+str(j)+'_'+str(i+1)+'.jpg'
      MB_img[:,:,i] = cv2.imread(img_path6, cv2.IMREAD_GRAYSCALE)  # stacking up images into the array
  MB_matrix = np.zeros((MB_img[:,:,0].size,n_bands))
  for i in range(n_bands):
      MB_array = MB_img[:,:,i].flatten()  # covert 2d to 1d array 
      MB_arrayStd = (MB_array - MB_array.mean())/MB_array.std()  # Standardize each variable 
      MB_matrix[:,i] = MB_arrayStd
  MB_matrix.shape
  np.set_printoptions(precision=3)
  cov = np.cov(MB_matrix.transpose())

  # Eigen Values
  EigVal,EigVec = np.linalg.eig(cov)

  # print("Eigen values:\n\n", EigVal,"\n")
  # Ordering Eigen values and vectors
  order = EigVal.argsort()[::-1]
  EigVal = EigVal[order]
  EigVec = EigVec[:,order]
  PC = np.matmul(MB_matrix,EigVec)   #cross product
  PC_2d = np.zeros((img_shape[0],img_shape[1],n_bands))
  for i in range(n_bands):
      PC_2d[:,:,i] = PC[:,i].reshape(-1,img_shape[1])

  # narmalizing between 0 to 255
  PC_2d_Norm = np.zeros((img_shape[0],img_shape[1],n_bands))
  for i in range(n_bands):
      PC_2d_Norm[:,:,i] = cv2.normalize(PC_2d[:,:,i],  np.zeros(img_shape),0,255 ,cv2.NORM_MINMAX)
  bgr_image = cv2.merge([PC_2d_Norm[:,:,0], PC_2d_Norm[:,:,1], PC_2d_Norm[:,:,2]]) # combine first three channels
  rgb_image = cv2.cvtColor(bgr_image.astype(np.uint8), cv2.COLOR_BGR2RGB) # convert from BGR to RGB and cast to uint8
  cv2.imwrite('new_data/pca_result'+str(j-10)+'.jpg', rgb_image)
  # for i in range(1,6):
    # os.remove("intermediate/output_"+str(j)+"_"+str(i)+".jpg")
print(cnt)