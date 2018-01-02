## Writeup Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Camera_Cali.png "Cam Calib"
[image2]: ./output_images/Straight_Ori_vs_Undist.JPG "Straight_Ori_vs_Undist"
[image3]: ./output_images/Test2_ColorBin_vs_CombThresh.jpg "Binary Example"
[image4]: ./output_images/Straight_Undist_vs_Warped.JPG "Warp Example"
[image5]: ./output_images/Test2_CombBin_vs_WarpedBin.jpg "Warped Bin"
[image6]: ./output_images/sliding_window_curve.png "sliding window for curve"
[image7]: ./output_images/Four_Points.JPG "picking 4 points"
[image8]: ./output_images/Test2out.JPG "output example image"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### 1. Camera Calibration

#### To calibrate the carmera, we need to compute the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell 1 to 5 of the IPython notebook located in "./Pipeline.ipynb". It's specifically done by the following code:
```python
# Iterate through the images, get the camera calibration chessboard corners
images = glob.glob('camera_cal/calibration*.jpg')

# initialize arrays to store object point and image points
objpoints = []
imgpoints = []

objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

for fname in images:
    img = cv2.imread(fname) #read in each image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
    ret,corners = cv2.findChessboardCorners(gray,(9,6),None)
    if ret==True:
        objpoints.append(objp)
        imgpoints.append(corners)
        img = cv2.drawChessboardCorners(img,(9,6),corners,ret)
        plt.imshow(img)
```

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

An example of finding corners is here:

![alt text][image1]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. This is done in code cell 4. I applied this distortion correction to the test image `straight_lines2.jpg` using the `cv2.undistort()` function and obtained this result: 

![alt text][image2]

### Pipeline

#### 1. Perform perspective transform.

This is done in the code cell 7 to 9. First I picked 4 points in the test image `straight_lines2.jpg` and 4 corresponding points in the bird eye view image:

![alt text][image7]

Then I wrapped up them as source points and destination points in the warp function which is defined in cell 8:

```python
def warp(img):
    img_size = (img.shape[1],img.shape[0])
    # 4 source points
    src = np.float32(
    [[704,458],
     [1046,670],
     [276,670],
     [582,458]])
    # 4 desired points
    dst = np.float32(
    [[1030,0],
     [1030,720],
     [250,720],
     [250,0]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
```

Apply this function on the test image `straight_lines2.jpg` and get the following result:

![alt text][image4]

#### 2. Perform color transforms and gradients threshold methods to create a thresholded binary image.

I used a combination of S channel of HLS and X-gradient thresholds to generate a binary image (cells 10 through 16 in `Pipeline.ipynb`).  Here's an example of my output for this step. This is generated using `test_images/test2.jpg`

![alt text][image3]

#### 3. Perform warp function on the binary map before we can run the sliding window algorithm to find lane lines

The ultimate goal of this project is to find lane lines and determine their curve radius. To do this, first we need to perform perspective transform on the binary map image. The perspective transform function was already wrapped in the warp function. Run it on the binary thresholding result image and get the following result:

![alt text][image5]

#### 4. Use sliding window method to find lane line pixels and use `np.polyfit` to calculate curve radius.

This part is done in code cell 18. First, sliding window method is applied to search for lane line pixels. Then `np.polyfit` function is applied to perform curve-fitting on the lane lines. The lane line curve radius can be calculated after this.

The vehicle offset from lane center is calculated using the following code:

```python
    offset = 640 - 0.5*(left_marker_x+right_marker_x)
```

The calculated curve radius and vehicle offset are in pixels, I then use the following parameters to convert them to meters:

```python
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.66/770 # meters per pixel in x dimension
```

Here's an example of the curve fitting result image:

![alt text][image6]

#### 5. Plot the results (lane lines, plus curve radius and offset info as text) back down onto the road such that the lane area is identified clearly.

I implemented this step in this cell:

```python
def draw_lane_area(image_in_undist,warped_combo,left_fit,right_fit,Minv,ave_curv_rad_m,offset_m):
    # Create an image to draw the lines on, and then wrap it in a function
    warped = warped_combo
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Generate points for curve fitting
    ploty = np.linspace(0, warped_combo.shape[0]-1, warped_combo.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image_in_undist.shape[1], image_in_undist.shape[0])) 
    
    # Combine the result with the original image
    result = cv2.addWeighted(image_in_undist, 1, newwarp, 0.3, 0)
    
    # Write Radius and Offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'Curve radius: ' + '{:04.2f}'.format(ave_curv_rad_m) + 'm'
    cv2.putText(result, text, (40,70), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    direction = ''
    if offset_m > 0:
        direction = 'right'
    elif offset_m < 0:
        direction = 'left'
    abs_offset_m = abs(offset_m)
    text = 'Vehicle is ' + '{:04.3f}'.format(abs_offset_m) + 'm ' + direction + ' from lane center'
    cv2.putText(result, text, (40,120), font, 1.5, (200,255,155), 2, cv2.LINE_AA)
    
    return result
```

Here is an example of my result on a test image:

![alt text][image8]

---

#### 6. Pipeline output video.

Here's a [link to my video result](https://youtu.be/24lxecUt1AA)

---

### Discussion

#### 1. One problem that I encountered: processing time

One problem that I encountered while processing the project video is that the processing time is too long (18 minutes). This is because I did not use a less time-consuming lane finding algorithm and the lane class method, due to not having enough time to work on this project. I will try other motheds later.
