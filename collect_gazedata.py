import random
import cv2
import numpy as np
import os
import screeninfo
from threading import Thread

class WebcamVideoStream:
    def __init__(self, src, width, height):
        self._stream = cv2.VideoCapture(src)
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self._stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while True:
            if self.stopped:
                return
            
            (self.grabbed, self.frame) = self._stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def screen_plot(gaze_loc, image=None, window_name='.'):
    screen = screeninfo.get_monitors()[0]

    canvas = np.zeros((screen.height, screen.width, 3), dtype=np.float32)

    # Plot image on center of canvas
    if image is not None:
        image = image / 255
        image_loc = [int(0.5 * screen.height - image.shape[0] / 2),
                     int(0.5 * screen.width - image.shape[1] / 2)]
        canvas[image_loc[0]:image_loc[0] + image.shape[0],
               image_loc[1]:image_loc[1] + image.shape[1]] = image


    # plot gaze location
    gaze_x = int(screen.width - gaze_loc[0] * screen.width)
    gaze_y = int(screen.height - gaze_loc[1] * screen.height)
    cv2.circle(canvas, (gaze_x, gaze_y), 20, (0, 255, 0), -1)
    cv2.line(canvas, (gaze_x, gaze_y-10), (gaze_x, gaze_y+10), (0, 0, 255), 2)
    cv2.line(canvas, (gaze_x-10, gaze_y), (gaze_x+10, gaze_y), (0, 0, 255), 2)

    # Plot infromation as text below
    """
    text = 'Gaze Location (%.2f, %.2f) \n %s' % (gaze_loc[0], gaze_loc[1], extra_text)
    text_x = int(0.4 * screen.width)
    text_y = int(0.7 * screen.height)
    cv2.putText(canvas, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    """

    # create named window and center & fullscreen it
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen.x - 1, screen.y - 1)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                          cv2.WINDOW_FULLSCREEN)

    return canvas


def main():
    person = 'Tang_xin'
    root_dir = '/home/user/Documents/gaze'
    dataset_dir = root_dir + '/' + 'data' + '/' + 'train'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    captureImg_width = 640
    captureImg_height = 480
    window_name = 'screen'
    video_capture = WebcamVideoStream(0,
                                      width=captureImg_width,
                                      height=captureImg_height).start()
    count = 1
    while True:
        # Pick a random gaze location
        gaze_x = random.uniform(0, 1)
        gaze_y = random.uniform(0, 1)

        # plot gaze location onto screen
        canvas = screen_plot([gaze_x, gaze_y],
                             window_name=window_name)
        cv2.imshow(window_name, canvas)
        cv2.waitKey()
        cv2.destroyAllWindows()

        frame = video_capture.read()
        # plot recoreded image and gaze location
        canvas = screen_plot([gaze_x, gaze_y],
                             image=frame,
                             window_name=window_name)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey() & 0xFF
        cv2.destroyAllWindows()

        if key == ord('s'):
            print('Image skipped')
            continue
        else:
            print('Saved image %d' % count)
            filename = '%s_%.2f_%.2f.jpg' % (person, gaze_x, gaze_y)
            cv2.imwrite(os.path.join(dataset_dir, filename), frame)
            count += 1

        if key == ord('q'):
            print('Quit gaze collection')
            break

    print('Data collection over, took %d images into %s' % (count, str(dataset_dir)))
    video_capture.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()