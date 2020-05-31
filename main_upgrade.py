import cv2
import dlib
import math

# 얼굴 랜드마크 검출 및 얼굴 회전 각도 계산
def detectLandmark():
    shape = predictor(gray, rect)
    
    leftear_area = shape.part(17)
    rightear_area = shape.part(22)

    degree_radian = math.atan2(shape.part(36).y-shape.part(45).y, shape.part(36).x-shape.part(45).x)
    degree = math.degrees(degree_radian)


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
            detectLandmark()




        cv2.imshow('snow app', img)
        if cv2.waitKey(1)==27:
            break
    cap.release()

            


    