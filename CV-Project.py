import math
import os
import tkinter
from tkinter import *
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

projectGroup = '18L-0969, 18L-1082, 18L-1105, 18L-2131'


def create_log_filter():
    logFilter = np.zeros((5, 5), int)
    logFilter[0, 2] = logFilter[1, 1] = logFilter[1, 3] = logFilter[2, 0] = logFilter[2, 4] = logFilter[3, 1] = \
        logFilter[
            3, 3] = logFilter[4, 2] = -1
    logFilter[1, 2] = logFilter[2, 1] = logFilter[2, 3] = logFilter[3, 2] = -2
    logFilter[2, 2] = 16
    return logFilter


def get_image(filePath):
    image = cv2.imread(filePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    row, col, channels = image.shape

    if row > factor:
        image = cv2.resize(image, (int(factor * col / row), factor), cv2.INTER_AREA)
        if col > factor:
            image = cv2.resize(image, (factor, int(factor * row / col)), cv2.INTER_AREA)
    elif col > factor:
        image = cv2.resize(image, (factor, int(factor * row / col)), cv2.INTER_AREA)
        if row > factor:
            image = cv2.resize(image, (int(factor * col / row), factor), cv2.INTER_AREA)

    gray_scaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_scaled_image


def apply_log(image):
    row, column = image.shape

    tempImage = np.zeros((row + 4, column + 4), int)
    bufferImage = np.zeros((row, column), int)

    for i in range(0, row):
        for j in range(0, column):
            tempImage[i + 2, j + 2] = image[i, j]

    for i in range(0, row):
        for j in range(0, column):
            value = 0
            for k in range(0, 5):
                for l in range(0, 5):
                    value += tempImage[i + k, j + l] * logFilter[k, l]
            bufferImage[i, j] = value

    return bufferImage


def hough_line(binary_image):
    row, col = binary_image.shape
    theta = 180
    accumulator = np.zeros((row + col, theta), int)
    grid = []

    for y in range(0, row + col):
        temp = []
        for x in range(0, theta):
            u = []
            temp.append(u)
        grid.append(temp)

    for y in range(0, row):
        for x in range(0, col):
            if binary_image[y, x] == 255:
                for i in range(0, theta):
                    val = int(x * np.cos(i * math.pi / 180) + y * np.sin(i * math.pi / 180))
                    accumulator[val, i] += 1
                    temp = [y, x]
                    grid[val][i].append(temp)

    acceptedList = []
    threshold = np.unique(accumulator)
    # threshold = np.average(threshold)
    threshold = threshold[int(len(threshold) * 0.8)]

    for y in range(0, row + col):
        for x in range(0, theta):
            if accumulator[y, x] > threshold:
                temp = [y, x]
                acceptedList.append(temp)

    # plt.axes()
    # plt.xlim([0, col])
    # plt.ylim([0, row])
    # plt.xticks(np.arange(0, col, 10))
    # plt.yticks(np.arange(0, row, 10))

    line_coordinates = []
    for each in acceptedList:
        temp = grid[each[0]][each[1]]
        y1 = temp[0][0]
        y2 = temp[len(temp) - 1][0]
        x1 = temp[0][1]
        x2 = temp[len(temp) - 1][1]

        xs = [x1, x2]
        ys = [y1, y2]
        temp2 = [xs, ys]
        line_coordinates.append(temp2)
    #     plt.plot(xs, ys)
    #
    # plt.show()
    tempImage = np.zeros((row, col), int)
    for each in line_coordinates:
        tempImage = cv2.line(tempImage, (each[0][0], each[1][0]), (each[0][1], each[1][1]), (255, 255, 255), 1)

    return tempImage


def draw_line(y, x):
    theeta = np.arange(-10, 10, 0.1)
    a = 4
    try:
        y *= 0.1
        x *= 0.1
    except:
        print(x)
    b1 = np.multiply(-x, -a) + y
    b2 = np.multiply(-x, a) + y

    temp = 'X: ' + str(x * 10) + ' Y: ' + str(y * 10) + ' (b1,b2) = (' + str(b1) + ', ' + str(b2) + ')'
    print(temp)
    plt.plot([-a, a], [b1, b2], )
    # val1 = x * np.cos(theeta)
    # val2 = y * np.sin(theeta)
    # row = np.add(val1,val2)
    # plt.plot(theeta,row)
    #
    # return theeta, row


def hough_circle(image):
    rows, cols = image.shape

    # 3d accumulator a,b,r
    max_radius = int(pow(pow(cols, 2) + pow(rows, 2), 0.5))  # max radius
    accumulator = np.zeros((cols, rows, max_radius), int)

    for y in range(0, rows):
        for x in range(0, cols):
            if image[y, x] == 255:
                for a in range(0, cols):  # a is in range cols that is x axis
                    for b in range(0, rows):  # b is in range rows that is y axis
                        r = int(
                            pow(pow(x - a, 2) + pow(y - b, 2), 0.5))  # circle equation (x - a)^2 + ( y - b) ^2 = r^2
                        accumulator[a, b, r] += 1

        if y % 20 == 0:
            print(int(y / rows * 50), ' %')

    threshold = np.unique(accumulator)
    threshold = threshold[int(len(threshold) * 0.8)]

    Accepted = []
    for a in range(0, cols):
        for b in range(0, rows):
            for r in range(0, max_radius):
                if accumulator[a, b, r] > threshold:
                    temp = [a, b, r]
                    flag = True
                    for each in Accepted:
                        if abs(each[2] - r) < 20 and pow(pow(each[0] - a, 2) + pow(each[1] - b, 2), 0.5) < 20:
                            flag = False
                            break

                    if flag:
                        Accepted.append(temp)
        if a % 20 == 0:
            print(int(a / cols * 50) + 50, ' %')

    # fig, ax = plt.subplots()
    # circles = []

    # plt.axes()
    # plt.xlim([0, cols])
    # plt.ylim([0, rows])
    # plt.xticks(np.arange(0, cols, 10))
    # plt.yticks(np.arange(0, rows, 10))

    # colors = ['red', 'green', 'blue', 'yellow']
    # for i in range(len(Accepted)):
    #     # print(Accepted[i])
    #     a = Accepted[i][0]
    #     b = Accepted[i][1]
    #     r = Accepted[i][2]
    #
    #     circle = plt.Circle((a, b), radius=r, color=colors[i % 4])
    #     circles.append(circle)
    #     plt.gca().add_patch(circle)
    #
    # plt.show()
    # Coordinates stored in Acccepted
    tempImage = np.zeros((rows, cols), int)
    for each in Accepted:
        tempImage = cv2.circle(tempImage, (each[0], each[1]), each[2], (255, 255, 255), 1)
    return tempImage


def create_GUI(root):
    root.wm_minsize(550, 200)
    root.title('CV-Project-18L-0969, 18L-1082, 18L-1105, 18L-2131')

    sourceFrame = Frame(root)
    sourceFrame.pack()
    sourceFrame.place(y=20)

    Label(sourceFrame, text='Click to Browse Image:', font='Georgia 11').grid(row=0, column=0)
    Label(sourceFrame, bd=5, width=40, bg='white', textvariable=filePath).grid(row=1, column=1)
    ttk.Button(sourceFrame, text="Browse", command=open_file).grid(row=1, column=3)

    B = Button(root, text="Calculate", command=performTask, bg='#cfe2e3', height=2, width=10, relief=RAISED,
               font='Herculaneum 10')
    B.pack()
    B.place(y=90, x=270)


def open_file():
    file = filedialog.askopenfile(mode='r', filetypes=[('Image Files', '*.*')])
    if file:
        filePath.set(os.path.abspath(file.name))


def binaryEdgeImage(image):
    print(image.shape)
    row, col = image.shape
    temp2 = np.zeros((row, col), int)
    boundary = 5

    for i in range(boundary, row - boundary):
        for j in range(boundary, col - boundary):
            if image[i, j] > 200:
                temp2[i, j] = 255

    return temp2


def andImages(tempImage, image):
    rows, cols = image.shape
    newImage = np.copy(tempImage)

    for i in range(0, rows):
        for j in range(0, cols):
            newImage[i, j] = tempImage[i, j] & image[i, j]

    return newImage


def orImages(tempImage, image):
    rows, cols = image.shape
    newImage = np.copy(tempImage)

    for i in range(0, rows):
        for j in range(0, cols):
            newImage[i, j] = tempImage[i, j] | image[i, j]

    return newImage


def performTask():
    try:
        gray_scaled_image = get_image(filePath.get())
        #LoG Filter
        edgeImage = apply_log(gray_scaled_image)
        #Thresholding
        binaryImage = binaryEdgeImage(edgeImage)
        #binaryImage = cv2.Canny(gray_scaled_image, 100, 250)

        image_line = hough_line(binaryImage) #Get Line Result
        image_circle = hough_circle(binaryImage)   #Get Circle Result
        houghAndLine = andImages(image_line, binaryImage) #Line AND Image Result
        houghAndCircle = andImages(image_circle, binaryImage) #Circle AND Image Result

        fig = plt.figure(figsize=(10, 10))

        fig.add_subplot(3, 3, 1)
        plt.imshow(gray_scaled_image, cmap='gray')
        plt.axis('off')
        plt.title("GrayScale Image:")

        fig.add_subplot(3, 3, 2)
        plt.imshow(edgeImage, cmap='gray')
        plt.axis('off')
        plt.title("LoG Image:")

        fig.add_subplot(3, 3, 3)
        plt.imshow(binaryImage, cmap='gray')
        plt.axis('off')
        plt.title("Binary Edge Image:")

        fig.add_subplot(3, 3, 4)
        plt.imshow(image_line, cmap='gray')
        plt.axis('off')
        plt.title("Hough Line:")

        fig.add_subplot(3, 3, 5)
        plt.imshow(image_circle, cmap='gray')
        plt.axis('off')
        plt.title("Hough Circle:")

        fig.add_subplot(3, 3, 6)
        plt.imshow(houghAndLine, cmap='gray')
        plt.axis('off')
        plt.title("Hough Line AND:")

        fig.add_subplot(3, 3, 7)
        plt.imshow(houghAndCircle, cmap='gray')
        plt.axis('off')
        plt.title("Hough Circle AND:")

        plt.show()

    except Exception as e:
        print(e)
        messagebox.showerror('Error', 'Invalid\\NO File Selected')


factor = 500
logFilter = create_log_filter()
root = Tk()
filePath = StringVar()
create_GUI(root)
root.mainloop()
