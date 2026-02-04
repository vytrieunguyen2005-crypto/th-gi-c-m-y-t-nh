import cv2
cam=cv2.VideoCapture(0)
while True :
    r,f = cam.read()
    if f is not None:
        cv2.imshow("abc", f )
    if cv2.waitKey(100) == ord("q"):
        break
cv2.destroyAllWindows()