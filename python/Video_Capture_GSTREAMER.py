#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
from datetime import datetime
import logging
from logging.handlers import SocketHandler


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import threading
import queue


# In[3]:


import cv2
from imutils.video import FPS


# In[4]:


def create_logger():
    # logging
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_filename = os.path.join(base_dir, "{}.log".format(date))
    # create logger with 'spam_application'
    logger = logging.getLogger("smart_vrt")
    logger.setLevel(CONFIG["logging"]["file_level"])
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(CONFIG["logging"]["file_level"])
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(CONFIG["logging"]["file_level"])
    # create a socket gandler to send to cutelog
    sh = SocketHandler('127.0.0.1', 19996)
    sh.setLevel(CONFIG["logging"]["file_level"])
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)-8s - %(threadName)-10s %(lineno)3d: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    sh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.addHandler(sh)
    
    return logger


# In[5]:


base_dir = 'data/'
CONFIG = {"logging" : {"file_level" : 'DEBUG'}}
logger = create_logger()


# In[6]:


class VideoCapture(threading.Thread):
    '''
        
        Thread responsável pela captura de vídeo.

    '''
    def __init__(self, silent=True):
        '''
        
            :video_file: Arquivo de video. Se for None pega os frames da câmera.
            :silent: Se for True, não printa o FPS.
            :output: Coloca os frames na fila q_raw.
        
        '''
        
        #Cria o objeto (thread) de captura de video.
        threading.Thread.__init__(self,name='VideoCapture')
        self.frame_queue = queue.Queue(2)

             
        self.fps = FPS()
        self.silent = silent
        self.input_width = 3280
        self.input_height = 2464
        gst_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM),"
        gst_str += "width=(int){}, height=(int){},format=(string)NV12,framerate=(fraction)10/1"
        gst_str += " ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR"
        gst_str += " ! appsink"
        
        gst_str = gst_str.format(self.input_width, self.input_height)
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        
    def run(self):
        try:
            # Start the device. This lights the LED if it's a camera that has one.
            self.capturing = True
            self.fps = FPS().start()

            if (self.cap.isOpened() == True):
                logger.info("Captura de video aberta com sucesso.")
            else:
                logger.error("Erro ao tentar abrir captura de video.")
            
            #Enquanto a função de captura estiver aberta e a bool capturing for True, 
            #captura o frame da câmera e coloca na fila q_prep_frame.
            while (self.cap.isOpened() and self.capturing):
                ret, frame = self.cap.read()
#                 self.frame_queue.put(frame)
                
#                 if self.frame_queue.full():
#                     self.frame_queue.get()

                # update the FPS counter
                self.fps.update()
    
                if not self.silent:
                    print("Frames Capturados: %.2f" % self.fps._numFrames)

        except Exception as inst:
            logger.error("{}".format(inst))
            
    def shutdown(self):
        self.capturing = False
        self.cap.release()
        self.fps.stop()
        logger.info("Captura de video finalizada.")
        logger.info("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        logger.info("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))


# In[7]:


vcap = VideoCapture()
vcap.start()
while vcap.fps._numFrames < 100:
    time.sleep(0.001)
vcap.shutdown()


# In[8]:


vcap.fps._numFrames


# In[ ]:




