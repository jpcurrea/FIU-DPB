###!/usr/bin/env python

import numpy as np
import serial as ser
from Tkinter import *
import tkFileDialog as fd
from threading import Timer
from serial.tools import list_ports
from time import sleep
import threading
import datetime as dt

ports = list_ports.comports()
port = ports[-1][0]

notes = []
minute = []

def getFile():
    root = Tk()
    root.geometry("0x0")
    root.title("file dialog")
    root.update()
    root.lift()
    f = fd.asksaveasfilename(filetypes=[('Numpy Binary', '*.npy')])
    root.destroy()
    return f

def printLight(port=port):
    light = ser.Serial(port)
    global close
    while (not close):
        print light.readline()
    light.close()


def readLight():
    light = ser.Serial(port)
    
    global close
    while (not close):
        rl = light.readline()
        print rl
        try: 
            global minute
            minute += [float(rl.replace('\r\n', ''))]
            # minute += [np.random.random_integers(0,17000)]
            # sleep(1)
        except:
            # pass
            global notes
            notes += [dt.datetime.now(), rl]
    light.close()
    

def recordMinute(time = 5):
    global fn
    sleep(time)
    while (not close):
        global minute
        tot = minute
        minute = []
        # f = open(fn, 'w')
        try:
            arr = np.load(fn)
            arr = np.insert(arr, -1, (dt.datetime.now(), np.mean(tot)))
        except:
            arr = np.array((dt.datetime.now(), np.mean(tot)), dtype=[('time', 'O'),('light', float)])
        # print arr
        np.save(fn, arr)
        # print s
        # f.write(s)
        sleep(time)
    # f.close()
        
def main(file_name):
    global close
    close = False

    global fn
    fn = file_name
    
    recorder = threading.Thread(target = recordMinute)
    reader = threading.Thread(target = readLight)

    recorder.start()
    reader.start()
    
    raw_input("Hit enter to stop recording.")
    close = True
    

# if (__name__ == "__main__"):
#     main()

