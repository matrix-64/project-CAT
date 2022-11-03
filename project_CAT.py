import numpy as np
import math
import cv2
import mediapipe as mp
import pyautogui as pg
import keyboard

pg.PAUSE = 0
pg.FAILSAFE = 0

from pynput.mouse import Button, Controller

class RemoteMouse:
    def __init__(self):
        self.mouse = Controller()

    def getPosition(self):
        return self.mouse.position

    def setPos(self, xPos, yPos):
        self.mouse.position = (xPos, yPos)
    def movePos(self, xPos, yPos):
        self.mouse.move(xPos, yPos)

    def click(self):
        self.mouse.click(Button.left)
    def doubleClick(self):
        self.mouse.click(Button.left, 2)
    def clickRight(self):
        self.mouse.click(Button.right)
    
    def down(self):
        self.mouse.press(Button.left)
    def up(self):
        self.mouse.release(Button.left)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def create_image(h, w, d):
    image = np.zeros((h, w,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image

def hshape(arr):
    r = arr[0]*16 + arr[1]*8 + arr[2]*4 + arr[3]*2 + arr[4]*1
    return r

SENSE = 2.3

cus_bef = [-1,-1] #커서(손가락)의 이전(before) 위치
cus_cur = [-1,-1] #커서(손가락)의 현재(current) 위치
finger = [[0,0],[0,0],[0,0],[0,0],[0,0]]
stdp_bef = [-1,-1]
stdp = [-1,-1]
Rclicking = False
Dclicking = False

def act_move():
    global cus_bef, cus_cur
    global finger
    
    cus_cur = [finger[1][0], finger[1][1]]
    if cus_bef == [-1,-1]: cus_bef = cus_cur
    cus_dif = [cus_cur[0]-cus_bef[0], cus_cur[1]-cus_bef[1]]
    if abs(cus_dif[0]) < 0.3 : cus_dif[0] = 0
    if abs(cus_dif[1]) < 0.3 : cus_dif[1] = 0
    
    moveX = math.sqrt(pow(abs(cus_dif[0]*3),3))*(1 if cus_dif[0]>0 else -1)
    moveY = math.sqrt(pow(abs(cus_dif[1]*3),3))*(1 if cus_dif[1]>0 else -1)
    
    return (moveX,moveY)

def act_subMove():
    global cus_bef, cus_cur
    dX,dY = act_move()
    mouse.movePos(SENSE * dX,SENSE * dY)
    cus_bef = cus_cur
    
def act_Rclick():
    global Rclicking
    if not Rclicking :
        Rclicking = True
        mouse.clickRight()

def act_Dclick():
    global Dclicking
    if not Dclicking :
        Dclicking = True
        mouse.doubleClick()

def act_scroll():
    global stdp_bef, stdp
    stdp_ydif = stdp[1]-stdp_bef[1]
    if abs(stdp_ydif) < 0.3 : stdp_ydif = 0
    
    moveY = SENSE * math.sqrt(pow(abs(stdp_ydif*3),3))*(1 if stdp_ydif>0 else -1)
    
    pg.scroll((-1)*int(moveY))

def action(sh):
    global cus_bef, cus_cur
    global finger
    global Rclicking, Dclicking

    if sh == 19 :
        act_subMove()
        mouse.down()
    else :
        mouse.up()
        if sh == 3 :
            act_subMove()
            return
        else : cus_bef = [-1,-1]
    
    if sh == 24 : act_Dclick()
    else : Dclicking = False
    
    if sh == 18 : act_Rclick()
    else : Rclicking = False
    
    if sh == 6 : act_scroll()

mouse = RemoteMouse()
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        
        height = image.shape[0]
        width = image.shape[1]
        depth = image.shape[2]
        
        dimage = create_image(height, width, depth) #바탕이 될 검은 이미지
        
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #BGR을 RGB로 변환
  
        results = hands.process(image) #손동작 인식
 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Opencv영상처리를 위해 다시 BGR로
 
        if results.multi_hand_landmarks: #result값이 정상인 경우에만 후속 작업
            for hls in results.multi_hand_landmarks:
                
                #손바닥의 가장 아랫점
                stdp = (hls.landmark[0].x * 100, hls.landmark[0].y * 100)
                
                #각 손가락의 끝점
                finger = [(hls.landmark[4].x * 100, hls.landmark[4].y * 100),
                          (hls.landmark[8].x * 100, hls.landmark[8].y * 100),
                          (hls.landmark[12].x * 100, hls.landmark[12].y * 100),
                          (hls.landmark[16].x * 100, hls.landmark[16].y * 100),
                          (hls.landmark[20].x * 100, hls.landmark[20].y * 100)]
                
                #각 손가락이 접혔는지를 판별(discriminate)할 기준점
                discr = [(hls.landmark[2].x * 100, hls.landmark[2].y * 100),
                          (hls.landmark[6].x * 100, hls.landmark[6].y * 100),
                          (hls.landmark[10].x * 100, hls.landmark[10].y * 100),
                          (hls.landmark[14].x * 100, hls.landmark[14].y * 100),
                          (hls.landmark[17].x * 100, hls.landmark[17].y * 100)]
                
                #손가락이 접혔는지를 나타내는 boolean 자료형의 리스트
                is_folded = [(finger[0][0] > discr[0][0]),
                             (finger[1][1] > discr[1][1]),
                             (finger[2][1] > discr[2][1]),
                             (finger[3][1] > discr[3][1]),
                             (finger[4][1] > discr[4][1])]
                
                hs = hshape(is_folded) #손모양(0~31까지의 정수)
                
                action(hs)
                stdp_bef = stdp
                
            cv2.putText(
                dimage, text='stdp=(%d,%d) d : %d %d %d %d %d' % (stdp[0],stdp[1],is_folded[0],is_folded[1],is_folded[2],is_folded[3],is_folded[4]), org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=255, thickness=3)
        
            mp_drawing.draw_landmarks(dimage, hls, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('CAT', dimage)
    
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyWindow('CAT')
