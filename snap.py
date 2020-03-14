#module to capture image from camera and save it as 'crop.png'
import cv2

def snap(filename='crop.png'):
    camera = cv2.VideoCapture(0)
    while(True):
        return_value, image = camera.read()
        cv2.imshow('capture',image)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.imwrite(filename, image)

    camera.release()
    del(camera)

if __name__ == "__main__":
    snap()