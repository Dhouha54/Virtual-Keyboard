import numpy as np
import cv2
import dlib
from math import hypot
import time
import pyglet
import autocomplete
import notification
import training


#load les images
image = cv2.imread('clavier.png')
image = cv2.resize(image, (200, 200))
imag_sugg = cv2.imread('sugg.png')
imag_sugg = cv2.resize(imag_sugg, (200, 200))
ensi = cv2.imread('ensilogo.jpg')
ensi = cv2.resize(ensi, (110, 90))
manouba = cv2.imread('manoubalogo.png')
manouba = cv2.resize(manouba, (80, 80))

#load sounds
sound = pyglet.media.load("sound.mp3", streaming=False)

#load camera and detector of face
cap = cv2.VideoCapture(0) 
detector = dlib.get_frontal_face_detector()  #detect face
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

#frame settings
affichage = np.zeros((580, 800, 3), np.uint8)
affichage[:] = (255, 255, 255)

letters= {"A": (0, 100),"B": (100, 100),"C": (200, 100), "D": (300, 100),"E": (400, 100),
"F": (500, 100),"G": (600, 100), "H": (700, 100),"I": (0, 200), "J": (100, 200),
"K": (200, 200), "L": (300, 200),"M": (400, 200), "N": (500, 200),"O": (600, 200),
"P": (700, 200), "Q": (0, 300), "R": (100, 300), "S": (200, 300), "T": (300, 300),
"U": (400, 300), "V": (500, 300), "W": (600, 300), "X": (700, 300), "Y": (0, 400),
"Z": (100, 400), "___": (200, 400), "<": (500, 400), "SMS": (600, 400),"understand": (15,20),"thanks": (275,20), "want": (535,20)}

suggestions = {"Yes": (15, 30),"No": (405, 30),"I'm hungry.": (15, 95), "I want to drink.": (405, 95),"I'm done.": (15, 160), 
"I'm do not understand.": (405, 160),"I want to watch TV.": (15, 225), "I want to go to the bathroom.": (405, 225),"I should take my medicine": (15, 290), "I don't know": (405, 290),
"Excuse me!": (15, 355), "I want to go outside": (405,355),"SMS": (15,420),"<": (405,420)}

keys_letters = list(letters.keys())
keys_suggestions = list(suggestions.keys())

# Counters
frames = 0 
sug_index = 0    
blinking_frame = 0
frames_to_blink = 10 #the number of frames to attend on blibking 
frames_active_letter = 20 #each nine frames light up difrrent letter 25
keys_number = 0
#text and keyboard setting
text=""
keyboard_selected = "left"
last_keyboard_selection = "left"
select_menu = True
keyboard_selection_frames = 0 

def add_words(text) :
    preduct = text.lower()
    possible = training.predict_next_word(preduct, 20)
    new_key = possible[0]
    value = letters.pop(list(letters.keys())[28])
    letters[new_key] = value
    print(len(keys_letters))
    print(list(letters.keys()))
    new_key1 = possible[1]
    value1 = letters.pop(list(letters.keys())[28])
    letters[new_key1] = value1
    print(len(keys_letters))
    print(list(letters.keys()))
    new_key2 = possible[2]
    value2 = letters.pop(list(letters.keys())[28])
    letters[new_key2] = value2
    print(len(keys_letters))
    print(list(letters.keys()))
def sms(text) :
    notification.envoyer_sms(text)
def complete(text) :
    possible = autocomplete.autocomplete(text.lower())
    new_key = possible[0]
    value = letters.pop(list(letters.keys())[29])
    letters[new_key] = value
    new_key1 = possible[1]
    value1 = letters.pop(list(letters.keys())[29])
    letters[new_key1] = value1
    new_key2 = possible[2]
    value2 = letters.pop(list(letters.keys())[29])
    letters[new_key2] = value2

def suggestion(index, text, light):
    #keys
    x= suggestions[text][0]
    y= suggestions[text][1]
    width = 380
    height = 63
    th = 3
    800
    if light is True :
        cv2.rectangle(affichage, (x + th ,y + th), (x + width - th ,y +  height - th), (57, 186, 113), -1)# -1 all the rectangle
        
    else : 
        cv2.rectangle(affichage, (x + th ,y + th), (x + width - th ,y +  height - th), (57, 186, 113), th)
    #text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    text_size = cv2.getTextSize(text, font_letter, 1.3, 4)[0] #((496, 38), 22)
    width_text , height_text = text_size[0] , text_size[1]
    text_x = int((width - width_text) /2) + x
    text_y = int((height + height_text) /2) + y
    
    cv2.putText(affichage, text, (text_x ,text_y), font_letter, 1.3, (0,0,0), 2)#text putting

def letter(letter_index, text, letter_light):
    #keys
    width =100
    
    x= letters[text][0]
    y= letters[text][1]
    
    
    height = 100
    th =2#thickness
    if letter_index == 26 :
        width =300
    elif letter_index == 28:
        width =200
    elif letter_index == 29 or letter_index == 30 or letter_index == 31:
        width = 250
        height = 60
        
    if letter_light is True:
        cv2.rectangle(affichage, (x + th, y+th), (x+width-th, y+height - th) ,(57, 186, 113),-1)
    else:
        cv2.rectangle(affichage, (x + th, y+th), (x+width-th, y+height - th) ,(57, 186, 113),th)
    
    #text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 5
    font_th=2
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    
    width_text , height_text= text_size[0],text_size[1]
    text_x= int((width - width_text) /2) + x
    text_y= int((height + height_text) /2) + y
    if letter_index == 29 or letter_index == 30 or letter_index == 31:
        font_scale = 1.7
        text_x= int(width/2) + x - 95
        text_y= 57
    cv2.putText(affichage, text, (text_x,text_y) ,font_letter, font_scale ,(0, 0, 0), font_th)
    

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2) , int((p1.y + p2.y)/2) #pixels can't be float
def get_blinking_ratio(eye_point , facial_landmarks) :
    left_point = (facial_landmarks.part(eye_point[0]).x,facial_landmarks.part(eye_point[0]).y)
    right_point = (facial_landmarks.part(eye_point[3]).x,facial_landmarks.part(eye_point[3]).y)
    hor_ligne = cv2.line(frame, left_point, right_point ,(0,255,0) , 1 ) #ligne horizentale
    
    center_top = midpoint(facial_landmarks.part(eye_point[1]),facial_landmarks.part(eye_point[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_point[5]),facial_landmarks.part(eye_point[4]))
    ver_ligne = cv2.line(frame, center_top, center_bottom ,(0,255,0) , 1 )
    
    ver_ligne_lenght = hypot((center_top[0] - center_bottom[0]),(center_top[1] - center_bottom[1]))
    hor_ligne_lenght = hypot((left_point[0] - right_point[0]),(left_point[1] - right_point[1]))
    if ver_ligne_lenght != 0:
        ratio = hor_ligne_lenght / ver_ligne_lenght
    else:
        ratio = None  # or assign any other default value that makes sense in your context
    #ratio = hor_ligne_lenght/ver_ligne_lenght #rapport est plus elevé quand yeux fermé 6/0.5=12 6/2=3
    return ratio
 
def get_gaze_ratio(eye_point, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_point[0]).x, facial_landmarks.part(eye_point[0]).y),
                               (facial_landmarks.part(eye_point[1]).x, facial_landmarks.part(eye_point[1]).y),
                               (facial_landmarks.part(eye_point[2]).x, facial_landmarks.part(eye_point[2]).y),
                               (facial_landmarks.part(eye_point[3]).x, facial_landmarks.part(eye_point[3]).y),
                               (facial_landmarks.part(eye_point[4]).x, facial_landmarks.part(eye_point[4]).y),
                               (facial_landmarks.part(eye_point[5]).x, facial_landmarks.part(eye_point[5]).y)], np.int32 )
    #cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2) #dessiner entour de yeux
    #creating a mask
    height, width, _ = frame.shape
    mask = np.zeros((height, width) , np.uint8)#the same size of the original frame : black screen
    cv2.polylines(mask, [left_eye_region], True, 255, 2) #entour de yeux
    cv2.fillPoly(mask,[left_eye_region],255 ) #partie de eye plein blanc 255
    eye = cv2.bitwise_and(gray, gray,  mask=mask) #appliquer mask sue eye
    
    
    min_x = np.min(left_eye_region[:, 0])#min of x
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])#min of y
    max_y = np.max(left_eye_region[:, 1]) 
    gray_eye = eye[min_y: max_y , min_x: max_x] #frame of eye
    
    #gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)#noir blanc image
    height , width = threshold_eye.shape
    left_side_threshold =  threshold_eye[0: height, 0: int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    
    right_side_threshold =  threshold_eye[0: height, int(width/2) :width] 
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white==0:
        gaze_ratio=0.5
    elif right_side_white==0:
        gaze_ratio=0.1
    else:
        gaze_ratio = left_side_white/ right_side_white
    return gaze_ratio
#580-800
def draw_menu() :
    #design pour page d'accueil
    cv2.line(affichage, (400, 260), (400, 700), (57, 186, 113), 3)
    cv2.putText(affichage, "EyeAssist", (300,70), font, 1, (57, 186, 113),3)
    fonte = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(affichage, "Welcome to EyeAssist, an application dedicated to providing medical assistance", (20,150), fonte, 0.5, (0, 0, 0),1)
    cv2.putText(affichage, "to those suffering from locked-in syndrome.", (20,180), fonte, 0.5, (0, 0, 0),1)
    affichage[10:100, 10:120] = ensi
    affichage[10:90, 690:770] = manouba
    affichage[300:500, 100:300] = image
    affichage[300:500, 500:700] = imag_sugg
    cv2.putText(affichage, "Look to the left ", (120,250), fonte, 0.5, (0,0,0),2)
    cv2.putText(affichage, "to access the virtual keyboard", (80,280), fonte, 0.5, (0,0,0),1)
    cv2.putText(affichage, "Look to the right", (520,250), fonte, 0.5, (0,0,0),2)
    cv2.putText(affichage, "to access the suggestion list", (480,280), fonte, 0.5, (0,0,0),1)


 
t=0
start = True
while True : 
    
    _, frame = cap.read()
    frame = cv2.resize(frame,None, fx=0.5 ,fy=0.5)
    affichage[:] = (255, 255, 255) #each time become white
    
    rows, cols, _ = frame.shape
    frames += 1 #augmenter nombre de frames
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    #draw a white space for loading bar
    frame[rows - 50: rows, 0:cols] = (255,255,255)
    if select_menu == True :
        draw_menu()
    #keyboard selected
    if keyboard_selected == "left":
        keys_set = list(letters.keys())
    else :
        keys_set = keys_suggestions
    
    active_sug = keys_set[sug_index]
    
    #face detection
    faces = detector(gray) #array where we have all the faces
    
    for face in faces :
        landmarks = predictor(gray , face)
        
        #eye detection
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio )/2
        
        if select_menu== True:
            text = ""
            #time.sleep(1)
            #cv2.putText(affichage, "start now", (50,50) ,font, 1 ,(255, 0, 0), 1)
            sug_index = 0 #initialiser conteur
            #gaze detection
            gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
            #cv2.putText(frame, str(gaze_ratio_left_eye), (150,150), font, 2, (0,0,255) , 3)
            gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)
            #cv2.putText(frame, str(gaze_ratio_right_eye), (150,250), font, 2, (0,0,255) , 3)
            gaze_ratio = (gaze_ratio_left_eye +gaze_ratio_right_eye) /2
            #0.1 #0.7
            if gaze_ratio < 0.8:
                cv2.putText(frame, "Left", (30,220), font, 1, (0,255,255),2)
                keyboard_selected = "left"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 30 :
                    select_menu = False
                    frames = 0 #set frames count to 0 when keyboard selected
                    keyboard_selection_frames = 0
                    keys_number = 32
                    
                
                #new_frame[:] = (0, 0, 255)
                #1.9
            #elif 0.1 <= gaze_ratio < 0.5:
                #cv2.putText(frame, "Center", (50,250), font, 3, (0,255,255))
            elif 0.8 <= gaze_ratio < 1.5:
                #new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "Right", (30,220), font, 1, (0,255,255),2)
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 30 :
                    select_menu = False
                    frames = 0 #set frames count to 0 when keyboard selected
                    keyboard_selection_frames = 0
                    keys_number = 14
                
                    
        else :
            cv2.rectangle(affichage, (10 ,510), (790 ,570), (0, 0, 0), -1)# -1 all the rectangle
        
            if blinking_ratio > 5 :
                cv2.putText(frame, "blinking", (50,150), font, 1, (0,255,0),3)
                blinking_frame += 1 #because each sug double and we have this counter to augmente the number while blinking
                frames -= 1 # frame stop counting ad sug doesn't change
                
                #typing sug
                if blinking_frame == frames_to_blink :
                    if (active_sug == "___") :
                        text+= " "
                        add_words(text)
                    elif( keys_set.index(active_sug) in {29,30,31} and (t==0)):
                        text=text[:-2];
                        text+= active_sug
                        t=1
                    elif active_sug == "SMS":
                        sms(text)
                    elif active_sug != "<":
                        text+= active_sug
                        complete(text)
                    else:
                        select_menu = True
                    sound.play()
                    time.sleep(1)
            else :
                blinking_frame =0
        
        if keyboard_selected == "left":
            keys_set = list(letters.keys())
        else :
            keys_set = keys_suggestions
        #diplay msg on the keyboard
        if select_menu is False :
            if frames == frames_active_letter :
                sug_index += 1
                frames = 0
            if sug_index == keys_number:
                sug_index=0
            for i in range(keys_number):
                if i == sug_index :
                    light = True
                else :
                    light = False
                if keyboard_selected == "right" :
                    suggestion(i , keys_set[i], light)
                else :
                    letter(i , keys_set[i], light)
       
        # Add the text to the frame
        cv2.putText(affichage, text, (60, 550), font, fontScale=1, color=(57, 186, 113), thickness=2)
        #cv2.putText(board, text, (10, 50), font, 0.5, (57, 186, 113), 2)
        
        
        
  

    cv2.imshow("Frame", frame)
    #cv2.imshow("new Frame", new_frame)
    cv2.imshow("Affichage", affichage)
   
    key = cv2.waitKey(1)
    if key == 27 :
        break
cap.release() #liberation de ressouce
cv2.destroyAllWindows()