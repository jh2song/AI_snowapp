import cv2
import dlib
import math
import numpy as np

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
#cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 320)



def Rabbit_Ears(img):
    img_leftear = cv2.imread('./rabbit_left_ear.png')
    img_rightear = cv2.imread('./rabbit_right_ear.png')
    img_leftear = cv2.resize(img_leftear, (71,136))
    img_rightear = cv2.resize(img_rightear, (71,136))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 얼굴 영역 검출
    faces = detector(gray)
    for rect in faces:

        leftOffsetY = 40
        leftOffsetX = 0
        rightOffsetY = 40
        rightOffsetX = 0

        # 얼굴 랜드마크 검출
        shape = predictor(gray, rect)
        
        leftear_area = shape.part(18)
        rightear_area = shape.part(23)

        faceLen = shape.part(16).x - shape.part(0).x      
        leftRate = (shape.part(27).x - shape.part(0).x) / faceLen
        rightRate = (shape.part(16).x - shape.part(27).x) / faceLen
        
        pt1 = (0,  136)
        pt2 = (71, 136)
        pt3 = (0, 0)
        pt4 = (71, 0)
        p1 = np.float32([pt1, pt2, pt3, pt4])
        
        pt1 = (0,  136)
        pt2 = (int(71*leftRate), 136)
        pt3 = (0 ,0)
        pt4 = (int(71*leftRate),0)
        p2 = np.float32([pt1, pt2, pt3, pt4])

        M =cv2.getPerspectiveTransform(p1,p2)
        img_leftear_tmp = cv2.warpPerspective(img_leftear, M,  (int(71*leftRate), 136));

        
        leftear_gray = cv2.cvtColor(img_leftear_tmp, cv2.COLOR_BGR2GRAY) 
        ret, leftear_mask = cv2.threshold(leftear_gray,1,255,cv2.THRESH_BINARY)
        leftear_mask_inv = cv2.bitwise_not(leftear_mask)
        
        pt1 = (0,  136)
        pt2 = (int(71*rightRate), 136)
        pt3 = (0 ,0)
        pt4 = (int(71*rightRate),0)
        p2 = np.float32([pt1, pt2, pt3, pt4])
        
        M =cv2.getPerspectiveTransform(p1,p2)
        img_rightear_tmp = cv2.warpPerspective(img_rightear, M,  (int(71*rightRate), 136));
        
        rightear_gray = cv2.cvtColor(img_rightear_tmp, cv2.COLOR_BGR2GRAY)
        ret, rightear_mask = cv2.threshold(rightear_gray,1,255,cv2.THRESH_BINARY)
        rightear_mask_inv = cv2.bitwise_not(rightear_mask)

        # 합성하는 토끼 귀 영역이 카메라를 벗어나는 예외 처리
        if (leftear_area.y - img_leftear.shape[0] - leftOffsetY < 0 or leftear_area.y - leftOffsetY > img.shape[0]):
            break
        
        if (leftear_area.x - img_leftear.shape[1] - leftOffsetX < 0 or leftear_area.x - leftOffsetX > img.shape[1]):
            break

        if (rightear_area.y - img_rightear.shape[0] - rightOffsetY < 0 or rightear_area.y - rightOffsetY > img.shape[0]):
            break

        if (rightear_area.x + rightOffsetX < 0 or rightear_area.x + img_rightear.shape[1] + rightOffsetX > img.shape[1]):
            break

        # 왼쪽 귀 비트마스킹
        leftroi = img[leftear_area.y - img_leftear.shape[0] - leftOffsetY : leftear_area.y - leftOffsetY, leftear_area.x - int(71*leftRate) - leftOffsetX : leftear_area.x - leftOffsetX]

        leftear_bg = cv2.bitwise_and(leftroi, leftroi, mask=leftear_mask_inv)
        leftear_fg = cv2.bitwise_and(img_leftear_tmp, img_leftear_tmp, mask=leftear_mask)
        dst = cv2.add(leftear_bg, leftear_fg)

        img[leftear_area.y - img_leftear.shape[0] - leftOffsetY : leftear_area.y - leftOffsetY, leftear_area.x - int(71*leftRate) - leftOffsetX : leftear_area.x - leftOffsetX] = dst
        #오른쪽귀
        rightroi = img[rightear_area.y - img_rightear.shape[0] - rightOffsetY : rightear_area.y - rightOffsetY, rightear_area.x + rightOffsetX : rightear_area.x + int(71*rightRate) + rightOffsetX]

        rightear_bg = cv2.bitwise_and(rightroi, rightroi, mask=rightear_mask_inv)
        rightear_fg = cv2.bitwise_and(img_rightear_tmp, img_rightear_tmp, mask=rightear_mask)
        dst = cv2.add(rightear_bg, rightear_fg)

        img[rightear_area.y - img_rightear.shape[0] - rightOffsetY : rightear_area.y - rightOffsetY, rightear_area.x + rightOffsetX : rightear_area.x + int(71*rightRate) + rightOffsetX] = dst
    #endfor
    return img