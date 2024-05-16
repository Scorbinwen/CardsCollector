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
        self.cards_dict = {}
        self.stock_dict = {}
        self.data_sequence = []

    def template_matching(self, image, template, scale):
        resized_template = cv2.resize(template, (int(template.shape[1] * scale), int(template.shape[0] * scale)))
        match = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, cols = np.where(match >= threshold)
        return cols

    def detect_player_cards(self, frame_region, scale):
        card_match_list = []
        # Load image and template
        for iter, file in enumerate(os.listdir(template_dir)):
            template_file = os.path.join(template_dir, file)
            template = cv2.imread(template_file)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            # Perform multi-scale template matching
            cols = self.template_matching(frame_region, template, scale)

            # Draw bounding boxes around the matched regions
            for col in cols:
                card_match_list.append([os.path.splitext(file)[0], col])

        card_match_list = sorted(card_match_list, key=lambda x: x[1])
        card_num_list = [t[0] for t in card_match_list]
        print("card_num_list", card_num_list)
        return card_num_list

    def preprocess_image(self, frame_region):
        (h, w) = frame_region.shape[:2]
        gry1 = cv2.resize(frame_region, (w * 2, h * 2))
        thr1 = cv2.threshold(gry1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return thr1

    def detect_player_stock(self, frame_region):
        pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

        processed_frame_region = self.preprocess_image(frame_region)
        # print("1 image", img)
        text = pytesseract.image_to_string(processed_frame_region, lang="eng", config="--psm 6 digits  -c tessedit_use_gpu=1")

        return text

    def update_if_change(self, ind, cur_frame, pre_frame):
        assert isinstance(self.monitor_cards_regions, list)
        cards_changed = False
        stock_changed = False
        # 检查是否更新局数了
        userid, scale, min_y, min_x, max_y, max_x = self.roll_region
        # 只检测其中一部分
        diff = cur_frame[min_x:max_x+1, min_y:max_y+1] - pre_frame[min_x:max_x+1, min_y:max_y+1]
        if not np.all(diff == 0):
            self.cards_dict = {}
            pre_frame = np.zeros_like(cur_frame)

        # 检查牌号更新
        for idx, region in enumerate(self.monitor_cards_regions):
            userid, scale,  min_y, min_x,  max_y, max_x = region
            # 只检测其中一部分
            diff = cur_frame[min_x:max_x+1, min_y:max_y+1] - pre_frame[min_x:max_x+1, min_y:max_y+1]
            if not np.all(diff == 0):
                player_cards = self.detect_player_cards(cur_frame[min_x:max_x+1, min_y:max_y+1], scale)
                # 有时候用户会没事手贱盖住一些牌,或者第一张牌被提示词遮挡无法识别出来的情况
                if userid in self.cards_dict and len(player_cards) > 0:
                    if len(self.cards_dict[userid]) <= len(player_cards):
                        cards_changed = True
                        for cards in player_cards:
                            self.cards_dict[userid].append(cards)
                elif len(player_cards) > 0:
                    self.cards_dict[userid] = player_cards
                    cards_changed = True
        # 检查注码更新
        for idx, region in enumerate(self.monitor_stock_regions):
            userid, scale, min_y, min_x,  max_y, max_x = region
            # 只检测其中一部分
            diff = cur_frame[min_x:max_x+1, min_y:max_y+1] - pre_frame[min_x:max_x+1, min_y:max_y+1]
            if not np.all(diff == 0):
                # player_stock is a digit
                player_stock = self.detect_player_stock(cur_frame[min_x:max_x+1, min_y:max_y+1])
                if player_stock is not None:
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if not ret:
                break

            cur_frame = frame
            if i == 0:
                pre_frame = np.zeros_like(cur_frame)
            self.update_if_change(i, cur_frame, pre_frame)
            pre_frame = cur_frame
            i = i + 1

if __name__ == "__main__":
    cards_collector = CardsCollector(video_path="origin_new.mp4")
    cards_collector.run_collect()
    print(cards_collector.data_sequence)
