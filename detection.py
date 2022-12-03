import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib

print("End to import Libraries")

my_image_dir = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject7\images\KakaoTalk_20221129_114027577_09.jpg"
img_bgr = cv2.imread(my_image_dir)
#img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_show = img_bgr.copy()


# https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html

detector_hog = dlib.get_frontal_face_detector()
dlib_rects = detector_hog(img_bgr, 1)
#  # (image, num of image pyramid)

print(dlib_rects)
for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()

model_dir = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject7\models\shape_predictor_68_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(model_dir)
print("End to upload weight")

list_landmarks = []

for dlib_rect in dlib_rects:
    points = landmark_predictor(img_bgr, dlib_rect)
        # 모든 landmark의 위치정보를 points 변수에 저장
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        # 각각의 landmark 위치정보를 (x,y) 형태로 변환하여 list_points 리스트로 저장
    list_landmarks.append(list_points)
        # list_landmarks에 랜드마크 리스트를 저장

print(len(list_landmarks[0]))

for landmark in list_landmarks:
    for point in landmark:
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)
            # cv2.circle: OpenCV의 원을 그리는 함수
            # img_show 이미지 위 각각의 point에
            # 크기가 2이고 (0, 255, 255)색으로 내부가 채워진(-1) 원을 그림
            # (마지막 인수가 자연수라면 그만큼의 두께의 선으로 원이 그려짐)

    # RGB 이미지로 전환
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
    # 이미지를 준비
plt.show()
    # 이미지를 출력

# zip() : 두 그룹의 데이터를 서로 엮어주는 파이썬의 내장 함수
# dlib_rects와 list_landmarks 데이터를 엮어 주었음
# dlib_rects : 얼굴 영역을 저장하고 있는 값
# → rectangles[[(345, 98) (531, 284)]]
# list_landmarks : 68개의 랜드마크 값 저장(이목구비 위치(x,y))

for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    # 얼굴 영역을 저장하고 있는 값과 68개의 랜드마크를 저장하고 있는 값으로 반복문 실행
    print (landmark[32]) # 코의 index는 30 입니다
    x = landmark[32][0] # 이미지에서 코 부위의 x값
    y = landmark[32][1] # 이미지에서 코 부위의 y값 - 얼굴 영역의 세로를 차지하는 픽셀의 수//2
    # y = landmark[30][1] - dlib_rect.height()//2
    w = h = dlib_rect.width() # 얼굴 영역의 가로를 차지하는 픽셀의 수 (531-345+1) → max(x) - min(x) +1(픽셀의 수 이기 때문에 1을 더해줌 → 픽셀 수는 점 하나로도 1이 됨)
    print (f'(x,y) : ({x},{y})')
    print (f'(w,h) : ({w},{h})')

sticker_dir = r"C:\Users\Jennie\Desktop\aiffel\Aiffel_MiniProject7\images\cat-whiskers.png"
img_sticker = cv2.imread(sticker_dir)
img_sticker = cv2.resize(img_sticker, (w, h), interpolation=cv2.INTER_CUBIC)
# 스티커 이미지 조정 → w,h는 얼굴 영역의 가로를 차지하는 픽셀의 수
# // cv2.resize(image객체 행렬, (가로 길이, 세로 길이))
print (img_sticker.shape) # 사이즈를 조정한 왕관 이미지의 차원 확인


# x,y,w,h 모두 위에서 반복문 안에서 지정해준 값임
# x는 이미지에서 코 부위의 x값 = 440
# y는 이미지에서 코 부위의 y값 = 286
# w는 얼굴 영역의 가로를 차지하는 픽셀의 수 = 322
# h는 얼굴 영역의 가로를 차지하는 픽셀의 수 = 322
refined_x = x - w//2 + 10 # 440 - (322//2) = 440-161 =279 (refined_x = x - w//2)
refined_y = y - h//2 + 2 # 89-322 = -233
# 원본 이미지에 스티커 이미지를 추가하기 위해서 x, y 좌표를 조정합니다.
# 이미지 시작점은 top-left 좌표이기 때문입니다.
# 즉, refined_x, refined_y값에서 왕관 이미지가 시작됨
print (f'(x,y) : ({refined_x},{refined_y})') # 음수 발생 : 이미지 범위를 벗어남
# 우리는 현재 이마 자리에 왕관을 두고 싶은건데, 이마위치 - 왕관 높이를 했더니 이미지의 범위를 초과하여 음수가 나오는 것
# opencv는 ndarray데이터를 사용하는데
# ndarray는 음수인덱스에 접근 불가하므로 스티커 이미지를 잘라 줘야 함

# 왕관 이미지가 이미지 밖에서 시작하지 않도록 조정이 필요함
# 좌표 순서가 y,x임에 유의한다. (y,x,rgb channel)
# 현재 상황에서는 -y 크기만큼 스티커를 crop 하고,
# top 의 x좌표와 y 좌표를 각각의 경우에 맞춰 원본 이미지의 경계 값으로 수정
# 음수값 만큼 왕관 이미지(혹은 추후 적용할 스티커 이미지)를 자른다.
if refined_x < 0: 
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
# 왕관 이미지를 씌우기 위해 왕관 이미지가 시작할 y좌표 값 조정
if refined_y < 0:
    img_sticker = img_sticker[-refined_y:, :]
    # refined_y가 -233이므로, img_sticker[233: , :]가 됨
    # (322,322, 3)에서 (286, 322, 3)이 됨 (322개 중에서 36개가 잘려나감)
    refined_y = 0

print (f'(x,y) : ({refined_x},{refined_y})')


# sticker_area는 원본이미지에서 스티커를 적용할 위치를 crop한 이미지 입니다.
# 예제에서는 (344,0) 부터 (344+187, 0+89) 범위의 이미지를 의미합니다.
# 좌표 순서가 y,x임에 유의한다. (y,x,rgb channel)
# img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
# img_show[0:0+286, 279:279+322]
# img_show[0:286, 279:601]
# 즉, x좌표는 279~601 / y좌표는 0~286가 됨
sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
# 왕관 이미지에서 사용할 부분은 0이 아닌 색이 있는 부분을 사용합니다.
# 왕관 이미지에서 0이 나오는 부분은 흰색이라는 뜻, 즉 이미지가 없다는 소리임.
# 현재 왕관 이미지에서는 왕관과 받침대 밑의 ------ 부분이 됨
# 그렇기 때문에 0인 부분(이미지가 없는 부분)은 제외하고 적용

# sticker_area는 원본 이미지에서 스티커를 적용할 위치를 미리 잘라낸 이미지
# 즉, 왕관 이미지에서 왕관 이미지가 없는 부분(왕관과 받침대 밑의 ------ 부분)은
# 원본 이미지에서 미리 잘라놓은 sticker_area(스티커 적용할 부분 만큼 원본 이미지에서 자른 이미지)를 적용하고,
# 나머지 부분은 스티커로 채워주면 됨

# np.where는 조건에 해당하는 인덱스만 찾아서 값을 적용하는 방법이다.
# 아래 코드에서는 img_sticker가 0일 경우(왕관 이미지에서 왕관 부분 제외한 나머지 이미지)에는
# sticker_area(원본 이미지에서 스티커를 적용할 위치를 미리 잘라낸 이미지)를 적용하고,
# 나머지 부분은 img_sticker(왕관 이미지)를 적용한다.
#img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    #np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
"""
img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
"""
# 왕관 이미지를 적용한 이미지를 보여준다.
# 얼굴 영역(7-3)과 랜드마크(7-4)를 미리 적용해놓은 img_show에 왕관 이미지를 덧붙인 이미지가 나오게 된다.)

sticker_width, sticker_height = img_sticker.shape[:2]
rotation = cv2.getRotationMatrix2D((sticker_width/2, sticker_height/2), -20, 0.9)
sticker_rotation = cv2.warpAffine(img_sticker, rotation, (sticker_width, sticker_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(sticker_rotation==255,sticker_area,sticker_rotation).astype(np.uint8)

plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()

# https://stackoverflow.com/questions/53106780/specify-background-color-when-rotating-an-image-using-opencv-in-python




# 위에서 설명했으므로 생략
# 왕관 이미지
sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
# img_bgr은 7-2에서 rgb로만 적용해놓은 원본 이미지이다. 
img_bgr[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(sticker_rotation==255,sticker_area,sticker_rotation).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) # rgb만 적용해놓은 원본 이미지에 왕관 이미지를 덮어 씌운 이미지가 나오게 된다.
plt.show()
