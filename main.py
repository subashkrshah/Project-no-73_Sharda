import matplotlib.pylab as plt
import cv2
import numpy as np

car_cascade = cv2.CascadeClassifier('carx.xml')

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 220, 0), thickness=4)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    #image = cv2.imread("road.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(image.shape)
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [# to be adjusted as per the image or road, i.e. till road's extreme vertices
        (0, height),
        (0.5*width , 0.5*height),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=1)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('car_view.mp4')

while(True):

    ret, frame = cap.read()
    frame = process(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#plt.imshow(canny_image)
#plt.imshow(cropped_image)
#plt.imshow(image)
#plt.imshow(image_with_lines)
#plt.show()
