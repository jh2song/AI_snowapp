import cv2
import dlib
import math

import cv2

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


if __name__ == '__main__':
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(0)

    img_leftear = cv2.imread('./rabbit_left_ear.png')
    img_rightear = cv2.imread('./rabbit_right_ear.png')

    img_leftear = cv2.resize(img_leftear, (70,130))
    img_rightear = cv2.resize(img_rightear, (70,130))
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print('no frame.')
            break
    
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 영역 검출
        faces = detector(gray)
        for rect in faces:
            # 얼굴 랜드마크 검출
            shape = predictor(gray, rect)
    
            leftear_area = shape.part(21)
            rightear_area = shape.part(22)
                                  
            # 회전 각도 계산
            degree_radian = math.atan2(shape.part(45).y-shape.part(36).y, shape.part(45).x-shape.part(36).x)
            degree = math.degrees(degree_radian)
            degree = 360 - degree

 


            rotated_left = rotate_image(img_leftear, degree)
            
            rotated_left_gray = cv2.cvtColor(rotated_left, cv2.COLOR_BGR2GRAY)
            ret, rotated_left_mask = cv2.threshold(rotated_left_gray, 1, 255, cv2.THRESH_BINARY)
            rotated_left_mask_inv = cv2.bitwise_not(rotated_left_mask)
            
            rotated_right = rotate_image(img_rightear, degree)
            rotated_right_gray = cv2.cvtColor(rotated_right, cv2.COLOR_BGR2GRAY)
            ret, rotated_right_mask = cv2.threshold(rotated_right_gray, 1, 255, cv2.THRESH_BINARY)
            rotated_right_mask_inv = cv2.bitwise_not(rotated_right_mask)
            
            y = leftear_area.y - rotated_left.shape[0]
            x = leftear_area.x - rotated_left.shape[1]
            h = rotated_left.shape[0]
            w = rotated_left.shape[1]
            
            if y<0:
                y=0
            if y+h>img.shape[0]:
                h=img.shape[0]-y
            if x<0:
                x=0
            if x+w>img.shape[1]:
                w=img.shape[1]-x 

            # 왼쪽 귀 비트마스킹
            leftroi = img[y:y+h, x:x+w]
            leftear_bg = cv2.bitwise_and(leftroi, leftroi, mask=rotated_left_mask_inv)
            leftear_fg = cv2.bitwise_and(rotated_left, rotated_left, mask=rotated_left_mask)
            dst = cv2.add(leftear_bg, leftear_fg)

            img[y:y+h, x:x+w] = dst

        cv2.imshow('snow app', img)
        if cv2.waitKey(1)==27:
            break
    cap.release()