import cv2
import numpy as np
import matplotlib.pyplot as plt

""" image 1
# Load the image
img0 = cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/dump_match_pairs/match_output/image1.jpg')

# Good 3D keypoints (from your previous results)
good_keypoints_3D = np.array([
    [0.0, 3.6, 0.7],
    [1.1, 2.1, 0.16],
    [-1.1, 2.1, 0.16],
    [1.7, 2.1, 0.06],
    [-1.7, 2.1, 0.06],
    [4.0, 0.08, 0.0],
    [-4.0, 0.08, 0.0],
    [-0.66, 2.4, -0.6],
    [0.66, 2.4, -0.6]
])

# Good 2D keypoints in Image 0 (from your previous results)
good_keypoints_2D = np.array([
    [319.0, 209.0],
    [276.0, 232.0],
    [363.0, 233.0],
    [252.0, 236.0],
    [387.0, 237.0],
    [177.0, 242.0],
    [466.0, 243.0],
    [343.0, 264.0],
    [293.0, 265.0]
])
"""
#image2
# Load the image
img0 = cv2.imread('/home/runbk0401/SuperGluePretrainedNetwork/assets/Ruun_images/cube/better_frame/0002.png')

# Good 3D keypoints (from your previous results)
good_keypoints_3D = np.array([
    [ 0.  ,    0.  ,    5.75  ],
        [-0.25 ,   0.25,    5.25  ],
        [ 0.25 ,  -0.25,    5.25  ],
        [ 0.25 ,   0.25,    5.25  ],
        [ 0.   ,   0.25,    5.1414],
        [-0.1414,  0.45,    5.    ],
        [ 0.1414,  0.25,    5.    ],
        [ 0.1414,  0.45,    5.    ],
        [ 0.    ,  0.25,    4.8586],
        [ 0.    ,  0.45,    4.8586],
        [ 0.25  ,  0.25,    4.75  ]
])

# Good 2D keypoints in Image 2 (from your previous results)
good_keypoints_2D = np.array([
    [545., 274.],
        [693., 479.],
        [401., 481.],
        [548., 508.],
        [624., 539.],
        [728., 600.],
        [582., 609.],
        [648., 623.],
        [623., 656.],
        [688., 671.],
        [550., 724.]
])

# Plot the keypoints on the image
for i, kp in enumerate(good_keypoints_2D):
    # Draw a circle for each keypoint in image0
    img0 = cv2.circle(img0, (int(kp[0]), int(kp[1])), 10, (0, 255, 0), 2)  # Green circle
    img0 = cv2.putText(img0, f'{i+1}', (int(kp[0]) + 15, int(kp[1])), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 0, 0), 2)  # Blue label with keypoint index

# Display the image with keypoints
plt.figure(figsize=(12.8, 9.6))  # (width, height) in inches, maintaining the 1280x960 aspect ratio
plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
plt.title('Good Keypoints on Image 2')
plt.show()
