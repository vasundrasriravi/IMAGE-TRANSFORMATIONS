# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import necessary libraries (NumPy, OpenCV, Matplotlib).
### Step2:
Read an image, convert it to RGB format, and display it using Matplotlib.Define translation parameters (e.g., shifting by 100 pixels horizontally and 200 pixels vertically).Perform translation using cv2.warpAffine().Display the translated image using Matplotlib.

### Step3:
Obtain the dimensions (rows, cols, dim) of the input image.Define a scaling matrix M with scaling factors of 1.5 in the x-direction and 1.8 in the y-direction.Perform perspective transformation using cv2.warpPerspective(), scaling the image by a factor of 1.5 in the x-direction and 1.8 in the y-direction.Display the scaled image using Matplotlib.

### Step4:
Define shear matrices M_x and M_y for shearing along the x-axis and y-axis, respectively.Perform perspective transformation using cv2.warpPerspective() with the shear matrices to shear the image along the x-axis and y-axis.Display the sheared images along the x-axis and y-axis using Matplotlib.

### Step5:
Define reflection matrices M_x and M_y for reflection along the x-axis and y-axis, respectively.Perform perspective transformation using cv2.warpPerspective() with the reflection matrices to reflect the image along the x-axis and y-axis.Display the reflected images along the x-axis and y-axis using Matplotlib.

### Step6:
Define an angle of rotation in radians (here, 10 degrees).Construct a rotation matrix M using the sine and cosine of the angle.Perform perspective transformation using cv2.warpPerspective() with the rotation matrix to rotate the image.Display the rotated image using Matplotlib.

### Step7:
Define a region of interest by specifying the desired range of rows and columns to crop the image (here, from row 100 to row 300 and from column 100 to column 300).Use array slicing to extract the cropped region from the input image.Display the cropped image using Matplotlib.

## Program:

### Developed By: VASUNDRA SRI R
### Register Number: 212222230168

## i)Image Translation
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("tran.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()
rows,cols,dim=input_image.shape
M = np.float32([[1, 0, 100],[0, 1, 200],[0, 0, 1]])

translated_image = cv2.warpPerspective(input_image, M, (cols, rows))
plt.axis('off')
plt.imshow(translated_image)
plt.show()
```
## ii) Image Scaling
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("tran.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

rows, cols, dim = input_image.shape 
M = np.float32([[1.5, 0, 0],[0, 1.8, 0],[0, 0, 1]])

scaled_img=cv2.warpPerspective (input_image, M, (cols*2, rows*2))
plt.imshow(scaled_img)
plt.show()
```
## iii)Image shearing
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("tran.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

M_x = np.float32([[1, 0.5, 0],[0, 1 ,0],[0,0,1]])
M_y =np.float32([[1, 0, 0],[0.5, 1, 0],[0, 0, 1]])

sheared_img_xaxis=cv2.warpPerspective(input_image,M_x, (int(cols*1.5), int(rows *1.5)))
sheared_img_yaxis = cv2.warpPerspective(input_image,M_y,(int(cols*1.5), int(rows*1.5)))

plt.imshow(sheared_img_xaxis)
plt.show()

plt.imshow(sheared_img_yaxis)
plt.show()
```
## iv)Image Reflection
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("tran.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

M_x= np.float32([[1,0, 0],[0, -1, rows],[0, 0, 1]])
M_y =np.float32([[-1, 0, cols],[ 0, 1, 0 ],[ 0, 0, 1 ]])
# Apply a perspective transformation to the image
reflected_img_xaxis=cv2.warpPerspective (input_image, M_x,(int(cols), int(rows)))
reflected_img_yaxis= cv2.warpPerspective (input_image, M_y, (int(cols), int(rows)))

                                         
plt.imshow(reflected_img_xaxis)
plt.show()

plt.imshow(reflected_img_yaxis)
plt.show()
```
## v)Image Rotation
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("tran.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
rotated_img = cv2.warpPerspective(input_image,M,(int(cols),int(rows)))

plt.imshow(rotated_img)
plt.show()
```
## vi)Image Cropping
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
input_image = cv2.imread("tran.jpg")
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(input_image)
plt.show()

cropped_img= input_image[100:300,100:300]

plt.imshow(cropped_img)
plt.show()

```
## Output:
## i)Image Translation

![Screenshot 2024-03-14 183832](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/922c3790-8d3c-4839-b09d-904797555be2)
![Screenshot 2024-03-14 183909](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/6abcfb41-cffb-48a8-80c7-aa90d3007809)

## ii) Image Scaling

![Screenshot 2024-03-14 183519](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/1988a5fc-408d-44b2-8760-bf4d28d105c5)
![Screenshot 2024-03-14 183538](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/3578ade7-2db5-470c-a7dd-09edd5d12352)

## iii)Image shearing

![Screenshot 2024-03-14 184034](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/efcc614a-1f9e-414e-9a17-baddade8576c)
![Screenshot 2024-03-14 184051](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/79208aec-3e45-4e67-992d-54a0bec19f19)
![Screenshot 2024-03-14 184106](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/a88a07d7-1f1d-435e-8727-65a82b409eaf)

## iv)Image Reflection

![Screenshot 2024-03-14 184327](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/f530d897-1a92-4086-974f-76d2f2fb45ee)
![Screenshot 2024-03-14 184342](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/807ed51d-28c8-44a3-89e4-d1d7b66b7cc9)
![Screenshot 2024-03-14 184356](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/7ae6ccad-197a-41a1-83fb-d3ef5bbb9772)

## v)Image Rotation

![Screenshot 2024-03-14 184547](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/27e32887-f918-4296-8848-54d4b8d0c966)
![Screenshot 2024-03-14 184604](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/54ff6188-bcf5-4bd1-b99b-a6aaed4f406d)

## vi)Image Cropping

![Screenshot 2024-03-14 184737](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/c0741ae1-0394-475e-afca-7e76cb68a1fc)
![Screenshot 2024-03-14 184756](https://github.com/vasundrasriravi/IMAGE-TRANSFORMATIONS/assets/119393983/adfee936-513b-4183-a766-5635a5dfcae4)

## Result: 
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
