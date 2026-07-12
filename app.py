import pygame, sys, os
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = load_model(os.path.join(BASE_DIR, "bestmodel.h5"))

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize pygame
pygame.init()

# Use standard built-in font
FONT = pygame.font.Font("freesansbold.ttf", 18)

DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
image_cnt = 1
number_xcord = []
number_ycord = []
PREDICT = True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        # Draw on the board when mouse is moving and writing is active
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            
            # Track coordinate points of the drawing
            number_xcord.append(xcord)
            number_ycord.append(ycord)
            
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True 

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            # Process the drawn digit if coordinates are captured
            if len(number_xcord) > 0 and len(number_ycord) > 0:
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                # Calculate bounding box coordinates around the drawn digit
                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(number_ycord[-1] + BOUNDARYINC, WINDOWSIZEY)

                number_xcord = []
                number_ycord = []

                # Extract the drawn pixel bounding box and free the screen lock
                pxarray = pygame.PixelArray(DISPLAYSURF)
                img_arr = np.array(pxarray)[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
                del pxarray

                # Threshold pixels to binary black/white format (0 or 255)
                img_arr = np.where(img_arr > 0, 255.0, 0.0)

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                    image_cnt += 1

                if PREDICT:
                    # Process and normalize image to fit MNIST dataset dimensions (28x28)
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0
                    
                    # Model expects shape (1, 28, 28, 1)
                    predictions = MODEL.predict(image.reshape(1, 28, 28, 1))
                    label = str(LABELS[np.argmax(predictions)])

                    # Render prediction text and position it next to the bounding box
                    textSurface = FONT.render(label, True, RED, WHITE)
                    textRecObj = textSurface.get_rect()
                    textRecObj.left, textRecObj.bottom = rect_max_x, rect_max_y

                    DISPLAYSURF.blit(textSurface, textRecObj)

        # Clear the screen when the 'n' key is pressed
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)
                
    pygame.display.update()