import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep
from PyQt5 import uic,QtWidgets
from PyQt5 import QtGui
from threading import Thread
import sympy
from sympy.plotting import*

App=QtWidgets.QApplication([])
PrincipalGUI = uic.loadUi("maskGUI.ui")


def vetor(xp1,yp1,xp2,yp2):
    x1 = xp1-xp2
    y1 = yp1-yp2
    return x1,y1

def ArqCos(vetor1, vetor2):
    #print(vetor1)
    #print(vetor2)
    cos = ((vetor1[0]*vetor2[0])+(vetor1[1]*vetor2[1]))/((math.sqrt((vetor1[0]**2)+(vetor1[1]**2)))*(math.sqrt((vetor2[0]**2)+(vetor2[1]**2))))
    arqcos = math.degrees(math.acos(cos))
    return arqcos

def show(img):
    try:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    except:
        pass
    plt.imshow(img)
    plt.show()

def linear(img):
    _,img_linear = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    return img_linear

def selec_cor(img,hL,sL,vL,hH,sH,vH):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lowColor = np.array([hL,sL,vL])
    highColor = np.array([hH,sH,vH])

    mask = cv2.inRange(hsv_img,lowColor,highColor)



    sleep(0.1)
    return mask


setlimits = [0,0,110,255,90,255]
def setLimitsOfTrackbar():
    return setlimits


#tamanho_mask = quantos pixeis brancos seguidos tem q achar para considerar uma borda da mascara
#min_negative = quantos pixeis pretos seguidos determinarão que os pontos brancos acabaram
def white_points_vertical(mask_white,img,tamanho_mask,min_Negative):
    altura,largura = mask_white.shape
    c = 0
    anti_c = 0
    list_x_white_s = []
    list_y_white_s = []
    list_y_white_i = []
    list_x_white_i = []
    for x in range(0, largura,10):
        for y in range(0,altura):
            c = 0
            if mask_white[y][x] == 255:
                if mask_white[y][x-15] == 255:
                    for i in range(0,70,7):
                        try:
                            if mask_white[y+i][x] == 255:
                                c+=1
                            else:
                                anti_c += 1
                                if anti_c > min_Negative:
                                    break
                        except:
                            break
                    if c > tamanho_mask:
                        list_x_white_s.append(x)
                        list_y_white_s.append(y)
                    break
    c = 0
    anti_c = 0
    for x in list_x_white_s:
        for y in range(altura-1,0,-1):
            c = 0
            if mask_white[y][x] == 255:
                if mask_white[y][x-15] == 255:
                    for i in range(0,70,7):
                        try:
                            if mask_white[y-i][x] == 255:
                                c+=1
                            else:
                                anti_c += 1
                                if anti_c > min_Negative:
                                    break
                        except:
                            break
                    if c > tamanho_mask:
                        list_x_white_i.append(x)
                        list_y_white_i.append(y)
                    break


    return list_x_white_s,list_x_white_i,list_y_white_s,list_y_white_i


def main():
    #img = cv2.imread("Reta2.jpg")
    cap = cv2.VideoCapture(1)
    classificadorVideoFace = cv2.CascadeClassifier('cascade//classifier//cascade.xml')
    global arq_cos_value
    global vetor_sup_module
    global vetor_inf_module
    global vetor_sup_position
    global scal

    ese = 0
    esd = 0
    eie = 0
    eid = 0

    while run:
        ese = 0
        esd = 0
        eie = 0
        eid = 0
        s, img = cap.read()
        altura,largura,rbg = img.shape

        altura = round(altura/2)
        largura = round(largura/2)
        dim = largura,altura
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        a,b,c,d,e,f = setLimitsOfTrackbar()
        mask_white = selec_cor(img,a,b,c,d,e,f)
        mask_white = cv2.resize(mask_white, dim, interpolation = cv2.INTER_AREA)
        print(scal)


        xs,xi,ys,yi = white_points_vertical(mask_white,img,8,2)



        try:
            p1 = (xs[round(len(xs)/vetor_sup_module+vetor_sup_position)],ys[round(len(xs)/vetor_sup_module+vetor_sup_position)])#pontos vertor lateral superior
            p2 = (xs[round(len(xs)/20*vetor_sup_module+vetor_sup_position)],ys[round(len(xs)/20*vetor_sup_module+vetor_sup_position)])

            p4 = (xi[round(len(xi)/vetor_inf_module+vetor_inf_position)],yi[round(len(xi)/vetor_inf_module+vetor_inf_position)])#pontos vetor lateral inferior
            p5 =(xi[round(len(xi)/20*vetor_inf_module+vetor_inf_position)],yi[round(len(xi)/20*vetor_inf_module+vetor_inf_position)])

            p7 = xs[round(len(xs)/2)],ys[round(len(xs)/2)]#pontos vertor vertical
            p8 = xi[round(len(xi)/2)],yi[round(len(xi)/2)]

            p10 = (xs[0],ys[0])#pontos laterais superiores
            p11 = (xs[len(xs)-1],ys[len(ys)-1])

            p13 = (xi[0],yi[0])#pontos laterais inferiores
            p14 =(xi[len(xi)-1],yi[len(yi)-1])

            #vetor da reta parealela superior completa da mascara para testar proporcionalidade
            vetor_Sup = vetor(xs[0],ys[0],xs[len(xs)-1],ys[len(ys)-1])
            modulo_Sup = math.sqrt(vetor_Sup[0]**2+vetor_Sup[1]**2)


            #vetores truncados para testar paralelismo
            vetor_Sup_trunc = vetor(xs[round(len(xs)/vetor_sup_module+vetor_sup_position)],ys[round(len(xs)/vetor_sup_module+vetor_sup_position)],xs[round(len(xs)/20*vetor_sup_module+vetor_sup_position)],ys[round(len(xs)/20*vetor_sup_module+vetor_sup_position)])
            vetor_Inf_trunc = vetor(xi[round(len(xi)/vetor_inf_module+vetor_inf_position)],yi[round(len(xi)/vetor_inf_module+vetor_inf_position)],xi[round(len(xi)/20*vetor_inf_module+vetor_inf_position)],yi[round(len(xi)/20*vetor_inf_module+vetor_inf_position)])

            #vetor vertical para testa proporcionalidade
            vetor_Vertical = vetor(xs[round(len(xs)/4)],ys[round(len(xs)/4)],xi[round(len(xi)/4)],yi[round(len(xi)/4)])
            modulo_Vertical = math.sqrt(vetor_Vertical[0]**2+vetor_Vertical[1]**2)

            vetor_hipotenisa = vetor(xs[0],ys[0],xi[len(xi)-1],yi[len(yi)-1])

            #testa se a mascara está reta, e na medida/tamanho certo
            if (ArqCos(vetor_Sup_trunc,vetor_Inf_trunc)>arq_cos_value or ((2*(modulo_Vertical))<(modulo_Sup)) or (math.sqrt(vetor_hipotenisa[0]**2+vetor_hipotenisa[1]**2)<(modulo_Vertical*1.28))):
                #caso estiver algo errado
                img = cv2.line(img,(p1),(p2),(0, 0, 255), thickness=2, lineType=8)
                img = cv2.line(img,(p4),(p5),(0, 0, 255), thickness=2, lineType=8)
                img = cv2.line(img,(p7),(p8),(0, 0, 255), thickness=2, lineType=8)

                img = cv2.line(img,(p10),(p11),(0, 0, 255), thickness=2, lineType=8)
                img = cv2.line(img,(p13),(p14),(0, 0, 255), thickness=2, lineType=8)
                PrincipalGUI.lbl_elasticos.setText(f"Elasticos ST: ")
                PrincipalGUI.lbl_img_elasticos.setPixmap(QtGui.QPixmap("imgs//ST.bmp"))
                PrincipalGUI.prop_or_des.setPixmap(QtGui.QPixmap("imgs//des.bmp"))
                PrincipalGUI.lbl_result.setPixmap(QtGui.QPixmap("imgs//resultruim.bmp"))

            else:
                #caso estiver td certo
                #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                #TESTE DE HAAR CASCADE
                detecta = classificadorVideoFace.detectMultiScale(mask_white,scaleFactor=1.3,minNeighbors=scal,minSize=(30, 30))
                for(x, y, l, a) in detecta:
                    cv2.rectangle(img, (x, y), (x + l, y + a), (255, 0, 0), 2)

                    if(x<(largura/2) and y<(altura/2)):
                        ese = 1
                    if(x>(largura/2) and y<(altura/2)):
                        esd = 1
                    if(x<(largura/2) and y>(altura/2)):
                        eie = 1
                    if(x>(largura/2) and y>(altura/2)):
                        eid = 1

                #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                #TESTANDO SE HA 4 PONTOS NA IMG COM ELASTICOS
                if((ese+esd+eie+eid)==4):
                    PrincipalGUI.lbl_elasticos.setText(f"Elasticos OK: {ese+esd+eie+eid}")
                    PrincipalGUI.lbl_img_elasticos.setPixmap(QtGui.QPixmap("imgs//acert.bmp"))
                    PrincipalGUI.lbl_result.setPixmap(QtGui.QPixmap("imgs//resultbom.bmp"))
                else:
                    PrincipalGUI.lbl_elasticos.setText(f"Elasticos BAD: {ese+esd+eie+eid}")
                    PrincipalGUI.lbl_img_elasticos.setPixmap(QtGui.QPixmap("imgs//falha.bmp"))
                    PrincipalGUI.lbl_result.setPixmap(QtGui.QPixmap("imgs//resultruim.bmp"))
                #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

                img = cv2.line(img,(p1),(p2),(255, 0, 0), thickness=2, lineType=8)
                img = cv2.line(img,(p4),(p5),(255, 0, 0), thickness=2, lineType=8)
                img = cv2.line(img,(p7),(p8),(0, 255, 0), thickness=2, lineType=8)

                img = cv2.line(img,(p10),(p11),(0, 255, 0), thickness=2, lineType=8)
                img = cv2.line(img,(p13),(p14),(0, 255, 0), thickness=2, lineType=8)

                PrincipalGUI.prop_or_des.setPixmap(QtGui.QPixmap("imgs//prop.bmp"))
        except Exception as e:
            print(e)

        cv2.imwrite("imgs//showimg.jpg",img)
        cv2.imwrite("imgs//showmask.jpg",mask_white)
        PrincipalGUI.L_img.setPixmap(QtGui.QPixmap("imgs//showimg.jpg"))
        PrincipalGUI.lblMask.setPixmap(QtGui.QPixmap("imgs//showmask.jpg"))



        if cv2.waitKey(1) % 0xFF == ord('q'): break


def huemin(value):
    setlimits[0] = value
    PrincipalGUI.lhmin.setText(str(value))

def huemax(value):
    setlimits[3] = value
    PrincipalGUI.lhmax.setText(str(value))

def satmin(value):
    setlimits[1] = value
    PrincipalGUI.lsmin.setText(str(value))

def satmax(value):
    setlimits[4] = value
    PrincipalGUI.lsmax.setText(str(value))

def valmin(value):
    setlimits[2] = value
    PrincipalGUI.lvmin.setText(str(value))

def valmax(value):
    setlimits[5] = value
    PrincipalGUI.lvmax.setText(str(value))

arq_cos_value = 5
def ark_cos_slider(value):
    global arq_cos_value
    arq_cos_value = value
    PrincipalGUI.arc_cos_lbl_value.setText(str(value))

vetor_sup_module = 7
def vetor_sup_module_def(value):
    global vetor_sup_module
    vetor_sup_module = value
    PrincipalGUI.vetor_sup_lbl_value.setText(str(value))

vetor_sup_position = 8
def vetor_sup_position_def(value):
    global vetor_sup_position
    vetor_sup_position = value-2
    PrincipalGUI.vetor_sup_position_lbl_value.setText(str(value))

vetor_inf_module = 7
def vetor_inf_module_def(value):
    global vetor_inf_module
    vetor_inf_module = value
    PrincipalGUI.vetor_inf_lbl_value.setText(str(value))

vetor_inf_position = 4
def vetor_inf_position_def(value):
    global vetor_inf_position
    vetor_inf_position = value-2
    PrincipalGUI.vetor_inf_position_lbl_value.setText(str(value))

scal = 1
def scale(value):
    global scal
    # if(value<11):
    #     value = 11
    scal = value
    PrincipalGUI.lbl_scale.setText(str(value))


def run_break():
    global run
    run = False


def run_init():
    global run
    run = True
    main()


PrincipalGUI.HueMin.valueChanged[int].connect(huemin)
PrincipalGUI.HueMax.valueChanged[int].connect(huemax)
PrincipalGUI.SatMin.valueChanged[int].connect(satmin)
PrincipalGUI.SatMax.valueChanged[int].connect(satmax)
PrincipalGUI.ValMin.valueChanged[int].connect(valmin)
PrincipalGUI.ValMax.valueChanged[int].connect(valmax)
PrincipalGUI.stopBTN.clicked.connect(run_break)
PrincipalGUI.initBTN.clicked.connect(run_init)
PrincipalGUI.arc_cos.valueChanged[int].connect(ark_cos_slider)
PrincipalGUI.vetor_sup.valueChanged[int].connect(vetor_sup_module_def)
PrincipalGUI.vetor_inf.valueChanged[int].connect(vetor_inf_module_def)
PrincipalGUI.vetor_sup_position.valueChanged[int].connect(vetor_sup_position_def)
PrincipalGUI.vetor_inf_position.valueChanged[int].connect(vetor_inf_position_def)
PrincipalGUI.slide_scale.valueChanged[int].connect(scale)
#fazer tmb para a relação da hipotenusa

PrincipalGUI.show()
App.exec()
