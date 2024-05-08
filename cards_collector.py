import cv2
import numpy as np
from config import *
import os
import pytesseract

class CardsCollector:
    def __init__(self, video_path="origin.mp4"):
        # Path to your video file
        self.video_path = video_path
        self.monitor_cards_regions = monitor_cards_regions
        self.monitor_stock_regions = monitor_stock_regions
        self.roll_region = roll_region
        # monitor_regions:[[userid, min_x, min_y, max_x, max_y], [], [], [], []]
        self.cards_dict = {}
        self.stock_dict = {}
        self.data_sequence = []

    def template_matching(self, image, template, scale):
        matches = []

        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        # print("image:", image.shape)
        # print("template", template.shape)
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        # may detect more than one matches
        loc = np.where(result >= threshold)

        # Draw rectangles around the matches
        for pt in zip(*loc[::-1]):
            max_loc = pt
            matches.append({
                'scale': scale,
                'location': max_loc,
            })

        return matches

    def detect_player_cards(self, frame_region, scale):
        # card_list = [(scale, row, col, card_num),... ]
        card_list = []
        # Load image and template
        for iter, file in enumerate(os.listdir(template_dir)):
            template_file = os.path.join(template_dir, file)
            template = cv2.imread(template_file)
            # Perform multi-scale template matching
            matches = self.template_matching(frame_region, template, scale)

            # Draw bounding boxes around the matched regions
            for match in matches:
                scale = match['scale']
                location = match['location']

                w = int(template.shape[1] * scale)
                h = int(template.shape[0] * scale)
                top_left = location
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame_region, top_left, bottom_right, (0, 255, 0), 5)
                row_num = (top_left[1] + bottom_right[1]) / 2 // grid * grid
                col_num = (top_left[0] + bottom_right[0]) / 2

                card_list.append(os.path.splitext(file)[0])
                # # Display the result
                # cv2.imwrite("result_{}.jpg".format(iter), image)
        return card_list

    def preprocess_image(self, frame_region):
        gry1 = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        (h, w) = gry1.shape[:2]
        gry1 = cv2.resize(gry1, (w * 2, h * 2))
        # gry1 = gry1[30:(h * 2), w + 50:w * 2]
        thr1 = cv2.threshold(gry1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thr1

    def detect_player_stock(self, frame_region):
        pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

        processed_frame_region = self.preprocess_image(frame_region)
        # print("1 image", img)
        text = pytesseract.image_to_string(processed_frame_region, lang="eng", config="--psm 6 digits  -c tessedit_use_gpu=1")
        print("text", text)
        assert text != ""
        return int(text)

    def update_if_change(self, cur_frame, pre_frame):
        assert isinstance(self.monitor_cards_regions, list)
        cards_changed = False
        stock_changed = False
        # 检查是否更新局数了
        userid, scale, min_y, min_x, max_y, max_x = self.roll_region
        # 只检测其中一部分
        diff = cur_frame[min_x:max_x+1, min_y:max_y+1] - pre_frame[min_x:max_x+1, min_y:max_y+1]
        if np.all(diff == 0):
            self.cards_dict = {}
            stock_dict = {}
            pre_frame = np.zeros_like(cur_frame)

        # 检查牌号更新
        for region in self.monitor_cards_regions:
            userid, scale,  min_y, min_x,  max_y, max_x = region
            # 只检测其中一部分
            # print("cur_frame", cur_frame.shape)
            # print("cord:", min_y, min_x,  max_y, max_x)
            # print("region", cur_frame[min_x:max_x+1, min_y:max_y+1].shape)
            diff = cur_frame[min_x:max_x+1, min_y:max_y+1] - pre_frame[min_x:max_x+1, min_y:max_y+1]
            if np.all(diff == 0):
                player_cards = self.detect_player_cards(cur_frame[min_x:max_x+1, min_y:max_y+1], scale)
                print(player_cards)
                # 有时候用户会没事手贱盖住一些牌
                if userid in self.cards_dict and len(self.cards_dict[userid]) < len(player_cards):
                    self.cards_dict[userid] = player_cards
                    cards_changed = True
        # 检查注码更新
        for region in self.monitor_stock_regions:
            userid, scale, min_y, min_x,  max_y, max_x = region
            # 只检测其中一部分
            diff = cur_frame[min_x:max_x+1, min_y:max_y+1] - pre_frame[min_x:max_x+1, min_y:max_y+1]
            if np.all(diff == 0):
                # player_stock is a digit
                player_stock = self.detect_player_stock(cur_frame[min_x:max_x+1, min_y:max_y+1])
                self.stock_dict[userid] = player_stock
                stock_changed = True
        if cards_changed:
            self.data_sequence.append(self.cards_dict)
        if stock_changed:
            self.data_sequence.append(self.stock_dict)

    def run_collect(self):
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        i = 0
        # Loop through the video frames
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            # if i == 30:
            #     cv2.imwrite("std_img.jpg", frame)
            #     break
            # Check if the frame was read successfully
            if not ret:
                break
            cur_frame = frame
            if  i == 0:
                pre_frame = np.zeros_like(cur_frame)
            self.update_if_change(cur_frame, pre_frame)
            pre_frame = cur_frame
            # Display the frame
            i = i + 1

if __name__ == "__main__":
    cards_collector = CardsCollector()
    cards_collector.run_collect()
    print(cards_collector.data_sequence)
