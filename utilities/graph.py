import matplotlib.pyplot as plt
import numpy as np
import sys
from os import listdir
from os.path import isdir, join

def plotAll(mypath, flip):
    
    onlyfiles = [f for f in listdir(mypath)  if isdir(join(mypath, f))]
    control = 0 
    for i in onlyfiles:
        print(mypath+i)
        x,y,text,flip,e= plotMultiple(mypath+i, 1)
        if text[4] != '64':
            #plt.errorbar(x, y, e, label=text[2:5], linestyle='None', marker='^')            
            plt.plot(x,y, label=text[2:5])
        elif text[2] == '0.00025' and text[3] == '0.99' and text[4] == '64' and text[5] == '25' and control == 0:
            #plt.errorbar(x, y, e, label=text[2:5], linestyle='None', marker='^')           
            plt.plot(x,y, label=text[2:5])
            control = 1
    plt.legend(loc="lower right")

    titleText = text[1] + " " + "Batch Size" + " " + flip
    plt.suptitle(mypath + " " + titleText , fontsize=10)
    #plt.show()
    plt.savefig(mypath+titleText+'_line.png', bbox_inches='tight')
    plt.clf()
    onlyfiles = [f for f in listdir(mypath)  if isdir(join(mypath, f))]
    control = 0 
    for i in onlyfiles:
        print(mypath+i)
        x,y,text,flip,e= plotMultiple(mypath+i, 1)
        if text[2] != '0.00025':
            #plt.errorbar(x, y, e, label=text[2:5], linestyle='None', marker='^')            
            plt.plot(x,y, label=text[2:5])
        elif text[2] == '0.00025' and text[3] == '0.99' and text[4] == '64' and text[5] == '25' and control == 0:
            #plt.errorbar(x, y, e, label=text[2:5], linestyle='None', marker='^')           
            plt.plot(x,y, label=text[2:5])
            control = 1
    plt.legend(loc="lower right")

    titleText = text[1] + " " + "Learning Rate" + " " + flip
    plt.suptitle(mypath + " " + titleText , fontsize=10)
    #plt.show()
    plt.savefig(mypath+titleText+'_line.png', bbox_inches='tight')
    plt.clf()
    onlyfiles = [f for f in listdir(mypath)  if isdir(join(mypath, f))]
    control = 0 
    for i in onlyfiles:
        print(mypath+i)
        x,y,text,flip,e= plotMultiple(mypath+i, 1)
        if text[3] != '0.99':
            #plt.errorbar(x, y, e, label=text[2:5], linestyle='None', marker='^')            
            plt.plot(x,y, label=text[2:5])
        elif text[2] == '0.00025' and text[3] == '0.99' and text[4] == '64' and text[5] == '25' and control == 0:
            #plt.errorbar(x, y, e, label=text[2:5], linestyle='None', marker='^')           
            plt.plot(x,y, label=text[2:5])
            control = 1
    plt.legend(loc="lower right")

    titleText = text[1] + " " + "Discount Factor" + " " + flip
    plt.suptitle(mypath + " " + titleText , fontsize=10)
    #plt.show()
    plt.savefig(mypath+titleText+'_line.png', bbox_inches='tight')
    plt.clf()

def plotMultiple(timestr, flip):

    x = []
    y = []
    e = []
    timessplit = timestr.split("/")[-1]

        
    flipFlag= flip
    flag = 0


    f = open("./"+timestr+"/logfile.txt", "r")


    resultArray = []

    counter = 1
    a= f.readlines()
    text = a[0].split("'")
    text = [text[1],text[3],text[5],text[7],text[9],text[11]]
    print(text)
    for i in a:
        if "Results:" not in i:
            continue
        else:
            resultArray.append(i)

    for j in resultArray:
        if flip == 0:
            if flip == flag:
                split = j.split(" ")
                x.append(counter)
                y.append(float(split[2]))
                e.append(float(split[4].split(",")[0]))
                counter+=1
            flip += 1
            
        else:
            if flip == flag:
                split = j.split(" ")
                x.append(counter)
                y.append(float(split[2]))
                e.append(float(split[4].split(",")[0]))
                counter+=1

            flip = 0

    print(x,y,e)
    x = np.asarray(x)
    y = np.asarray(y)
    e = np.asarray(e)

    if flipFlag == 0:
        flip = "Test"
    else: 
        flip = "Train"

    return x,y,text,flip,e
  
def plotGrowth(timestr, flip):

    x = []
    y = []
    e = []
    timessplit = timestr.split("/")[-1]

        
    flipFlag= flip
    flag = 1


    f = open("./"+timestr+"/logfile.txt", "r")

    resultArray = []

    counter = 1
    a= f.readlines()
    text = a[0].split("'")
    text = text[1] + text[3] + text[5] + text[7] + text[9]
    
    for i in a:
        if "Results:" not in i:
            continue
        else:
            resultArray.append(i)

    for j in resultArray:
        if flip == 0:
            if flip == flag:
                split = j.split(" ")
                x.append(counter)
                y.append(float(split[2]))
                e.append(float(split[4].split(",")[0]))
                counter+=1
            flip += 1
            
        else:
            if flip == flag:
                split = j.split(" ")
                x.append(counter)
                y.append(float(split[2]))
                e.append(float(split[4].split(",")[0]))
                counter+=1

            flip = 0

    print(x,y,e)
    x = np.asarray(x)
    y = np.asarray(y)
    e = np.asarray(e)

    if flipFlag == 0:
        flip = "Test"
    else: 
        flip = "Train"

    plt.clf()
    plt.suptitle( text + " " + flip, fontsize=10)
    
    plt.errorbar(x, y, e, linestyle='None', marker='^')

    plt.savefig("./"+timestr+"/"+timessplit+"_"+str(flip)+'_std.png', bbox_inches='tight')
    print("./"+timestr+"/"+timessplit+"_"+str(flip)+'_std.png')
    plt.clf()
    plt.suptitle(text + " " + flip, fontsize=10)
    plt.plot(x, y)
    plt.savefig("./"+timestr+"/"+timessplit+"_"+str(flip)+'_line.png', bbox_inches='tight')



#plotAll("./Save/basic_turn/", 1)

