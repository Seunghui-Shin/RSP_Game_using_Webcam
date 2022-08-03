from tkinter import *
import random
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
from tkinter import messagebox

win = Tk()
win.geometry("1000x600")
win.title("가위바위보 게임")

max_num_hands = 1

# 해당 인덱스에 가위바위보 매칭
rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=max_num_hands,min_detection_confidence=0.5,min_tracking_confidence=0.5)


file = np.genfromtxt('gesture_train.csv', delimiter=',') # 해당 레이블에 대한 손가락 각도 파일
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label) #KNN을 이용해 학습


class MainWindow():
    #----------------
    def __init__(self, main):

        # canvas for image
        self.canvas1 = Canvas(main, width=400, height=300)
        self.canvas1.place(x=400, y=0)

        # images
        self.my_images1 = []
        self.my_images1.append(PhotoImage(file = "rock.png"))
        self.my_images1.append(PhotoImage(file = "scissors.png"))
        self.my_images1.append(PhotoImage(file = "paper.png"))
        self.my_image_number1 = 0

        # set first image on canvas
        self.image_on_canvas1 = self.canvas1.create_image(0, 0, anchor = NW, image = self.my_images1[self.my_image_number1])

        # button text
        self.text1 = StringVar()
        self.text1.set("0")
        self.text2 = StringVar()
        self.text2.set("0")
        self.text3 = StringVar()
        self.text3.set("0")
        self.text4 = StringVar()
        self.text4.set("0")
        self.text5 = StringVar()
        self.text5.set("0")
        self.text6 = StringVar()
        self.text6.set("0")

        # win,draw,lose text
        self.winuser = 0
        self.drawuser = 0
        self.loseuser = 0
        self.winpc = 0
        self.drawpc = 0
        self.losepc = 0

        #letter
        
        # button to change image
        self.button1 = Button(main, text="CAM ON", width=25, height=4, command = self.video_play)
        self.button2 = Button(main, text="GAME START", width=25, height=4, command = self.onButton)
        self.button3 = Button(main, text="GAME STOP", width=25, height=4, command = self.stop)
        self.button4 = Button(main, text="RESTART", width=25, height=4, command = self.restart)
        self.button6 = Button(main, text="SCORE BOARD", width=80, height=3)
        self.button7 = Button(main, text="승", width=15, height=3)
        self.button8 = Button(main, text="무", width=15, height=3)
        self.button9 = Button(main, text="패", width=15, height=3)
        self.button10 = Button(main, text="USER", width=15, height=3)
        self.button11 = Button(main, textvariable = self.text1, width=15, height=3)
        self.button12 = Button(main, textvariable = self.text2, width=15, height=3)
        self.button13 = Button(main, textvariable = self.text3, width=15, height=3)
        self.button14 = Button(main, text="COMPUTER", width=15, height=3)
        self.button15 = Button(main, textvariable = self.text4, width=15, height=3)
        self.button16 = Button(main, textvariable = self.text5, width=15, height=3)
        self.button17 = Button(main, textvariable = self.text6, width=15, height=3)
        self.button1.place(x=810, y=5)
        self.button2.place(x=810, y=80)
        self.button3.place(x=810, y=155)
        self.button4.place(x=810, y=230)
        self.button6.place(x=10, y=310)
        self.button7.place(x=160, y=380)
        self.button8.place(x=310, y=380)
        self.button9.place(x=460, y=380)
        self.button10.place(x=10, y=460)
        self.button11.place(x=160, y=460)
        self.button12.place(x=310, y=460)
        self.button13.place(x=460, y=460)
        self.button14.place(x=10, y=540)
        self.button15.place(x=160, y=540)
        self.button16.place(x=310, y=540)
        self.button17.place(x=460, y=540)

        # timer
        self.running = True
        
        # canvas for image
        self.canvas2 = Canvas(main, width=380, height=280)
        self.canvas2.place(x=610, y=310)

        # images
        self.my_images2 = []
        self.my_images2.append(PhotoImage(file = "three.png"))
        self.my_images2.append(PhotoImage(file = "two.png"))
        self.my_images2.append(PhotoImage(file = "one.png"))
        self.my_images2.append(PhotoImage(file = "go.png"))
        self.my_image_number2 = 0

        # set first image on canvas
        self.image_on_canvas2 = self.canvas2.create_image(0, 0, anchor = NW, image = self.my_images2[self.my_image_number2])
        
        # webcam
        self.frm = Frame(main,bg = "white",width=400,height=300)
        self.frm.place(x=0,y=0)
        self.lbl1 = Label(self.frm,width=400,height=300)
        self.lbl1.place(x=0,y=0)
        self.cap = cv2.VideoCapture(0) # VideoCapture 객체 정의
        self.idx = 0
    #----------------
    def stop(self):
        self.running = False
        self.my_image_number2 = 0
        self.canvas2.itemconfig(self.image_on_canvas2, image = self.my_images2[self.my_image_number2])
        
    def restart(self):
        self.running = True
        self.winuser = 0
        self.drawuser = 0
        self.loseuser = 0
        self.winpc = 0
        self.drawpc = 0
        self.losepc = 0
        self.text1.set(str(self.winuser))
        self.text2.set(str(self.drawuser))
        self.text3.set(str(self.loseuser))
        self.text4.set(str(self.winpc))
        self.text5.set(str(self.drawpc))
        self.text6.set(str(self.losepc))
    #----------------
    def onButton(self):
    
        if self.running:
            # next image
            self.my_image_number2 += 1

            # return to first image
            if self.my_image_number2 == len(self.my_images2):
                self.my_image_number1 = random.randint(0,2)
                self.canvas1.itemconfig(self.image_on_canvas1, image = self.my_images1[self.my_image_number1])
                self.my_image_number2 = 0
                if self.my_image_number1 == 0:
                    if self.idx == 0:
                        messagebox.showinfo("result", "비겼습니다!!")
                        self.drawuser +=1
                        self.drawpc +=1
                        self.text2.set(str(self.drawuser))
                        self.text5.set(str(self.drawpc))
                    elif self.idx == 5:
                        messagebox.showinfo("result", "이겼습니다!!")
                        self.winuser +=1
                        self.losepc +=1
                        self.text1.set(str(self.winuser))
                        self.text6.set(str(self.losepc))
                    else:
                        messagebox.showinfo("result", "졌습니다!!")
                        self.loseuser +=1
                        self.winpc +=1
                        self.text3.set(str(self.loseuser))
                        self.text4.set(str(self.winpc))
                elif self.my_image_number1 == 1:
                    if self.idx == 0:
                        messagebox.showinfo("result", "이겼습니다!!")
                        self.winuser +=1
                        self.losepc +=1
                        self.text1.set(str(self.winuser))
                        self.text6.set(str(self.losepc))              
                    elif self.idx == 5:
                        messagebox.showinfo("result", "졌습니다!!")
                        self.loseuser +=1
                        self.winpc +=1
                        self.text3.set(str(self.loseuser))
                        self.text4.set(str(self.winpc))                       
                    else:
                        messagebox.showinfo("result", "비겼습니다!!")
                        self.drawuser +=1
                        self.drawpc +=1
                        self.text2.set(str(self.drawuser))
                        self.text5.set(str(self.drawpc))                       
                else:
                    if self.idx == 0:
                        messagebox.showinfo("result", "졌습니다!!")
                        self.loseuser +=1
                        self.winpc +=1
                        self.text3.set(str(self.loseuser))
                        self.text4.set(str(self.winpc))
                    elif self.idx == 5:
                        messagebox.showinfo("result", "비겼습니다!!")
                        self.drawuser +=1
                        self.drawpc +=1
                        self.text2.set(str(self.drawuser))
                        self.text5.set(str(self.drawpc))
                    else:
                        messagebox.showinfo("result", "이겼습니다!!")
                        self.winuser +=1
                        self.losepc +=1
                        self.text1.set(str(self.winuser))
                        self.text6.set(str(self.losepc)) 
            # change image
            self.canvas2.itemconfig(self.image_on_canvas2, image = self.my_images2[self.my_image_number2])
        win.after(1000,self.onButton)
        
    #----------------
    
    def video_play(self):
        ret, img = self.cap.read() # 프레임이 올바르게 읽히면 ret은 True
        if not ret:
            self.cap.release() # 작업 완료 후 해제
            return
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3)) # 빨간 점들의 x,y,z 좌표 저장
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # 각 관절에 대한 벡터
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
                v = v2 - v1

                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # 아크코사인으로 각도를 구함
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle)


                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                self.idx = int(results[0][0])

                # 가위바위보 중 하나 글씨로 쓰기
                if self.idx in rps_gesture.keys():
                    cv2.putText(img, text=rps_gesture[self.idx].upper(), org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                    
                # 손가락 마디마디의 랜드마크
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
        

        imgframe = Image.fromarray(img) # Image 객체로 변환
        imgtk = ImageTk.PhotoImage(image=imgframe) # ImageTk 객체로 변환
        # OpenCV 동영상
        self.lbl1.imgtk = imgtk
        self.lbl1.configure(image=imgtk)
        self.lbl1.after(10, self.video_play)

    #----------------
#----------------------------------------------------------------------

MainWindow(win)
win.mainloop()