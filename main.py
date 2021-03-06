import cv2
import dlib
import math

# 얼굴 검출기와 랜드마크 검출기 생성
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
#cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, 480)
#cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, 320)

img_leftear = cv2.imread('./rabbit_left_ear.png')
img_rightear = cv2.imread('./rabbit_right_ear.png')



img_leftear = cv2.resize(img_leftear, (71,136))
img_rightear = cv2.resize(img_rightear, (71,136))
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print('no frame.');break
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

        leftear_area = shape.part(17)
        rightear_area = shape.part(22)
        
        rot_radian = math.atan2(shape.part(46).y-shape.part(37).y, shape.part(46).x-shape.part(37).x)
        degree = math.degrees(rot_radian)
        if(degree>0):
            leftOffsetX = int(2 * degree * -1)
            rightOffsetX = int(2 * degree)
        degree = 360 - degree
        
        

        rows, cols = img_leftear.shape[0:2]
        m45 = cv2.getRotationMatrix2D((cols/2, rows/2), degree, 1)
        img_leftear_tmp = cv2.warpAffine(img_leftear, m45, (cols, rows), None, cv2.INTER_LINEAR)
        leftear_gray = cv2.cvtColor(img_leftear_tmp, cv2.COLOR_BGR2GRAY) 
        ret, leftear_mask = cv2.threshold(leftear_gray,1,255,cv2.THRESH_BINARY)
        leftear_mask_inv = cv2.bitwise_not(leftear_mask)

        rows, cols = img_rightear.shape[0:2]
        m45 = cv2.getRotationMatrix2D((img_rightear.shape[1]/2, img_rightear.shape[0]/2), degree, 1)
        img_rightear_tmp = cv2.warpAffine(img_rightear, m45, (cols, rows), None, cv2.INTER_LINEAR)
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
        leftroi = img[leftear_area.y - img_leftear.shape[0] - leftOffsetY : leftear_area.y - leftOffsetY, leftear_area.x - img_leftear.shape[1] - leftOffsetX : leftear_area.x - leftOffsetX]
        
        leftear_bg = cv2.bitwise_and(leftroi, leftroi, mask=leftear_mask_inv)
        leftear_fg = cv2.bitwise_and(img_leftear_tmp, img_leftear_tmp, mask=leftear_mask)
        dst = cv2.add(leftear_bg, leftear_fg)
        
        img[leftear_area.y - img_leftear.shape[0] - leftOffsetY : leftear_area.y - leftOffsetY, leftear_area.x - img_leftear.shape[1] - leftOffsetX : leftear_area.x - leftOffsetX] = dst

        # 오른쪽 귀 비트마스킹
        rightroi = img[rightear_area.y - img_rightear.shape[0] - rightOffsetY : rightear_area.y - rightOffsetY, rightear_area.x + rightOffsetX : rightear_area.x + img_rightear.shape[1] + rightOffsetX]

        rightear_bg = cv2.bitwise_and(rightroi, rightroi, mask=rightear_mask_inv)
        rightear_fg = cv2.bitwise_and(img_rightear_tmp, img_rightear_tmp, mask=rightear_mask)
        dst = cv2.add(rightear_bg, rightear_fg)
        
        img[rightear_area.y - img_rightear.shape[0] - rightOffsetY : rightear_area.y - rightOffsetY, rightear_area.x + rightOffsetX : rightear_area.x + img_rightear.shape[1] + rightOffsetX] = dst


    cv2.imshow("snow app", img)
    if cv2.waitKey(1)== 27:
        break
cap.release()



