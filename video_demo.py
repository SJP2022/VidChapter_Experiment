import json
import os
import random
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
 
only_shot = False

# 自定义函数来添加换行
def wrap_text(draw, text, font, max_width):
    lines = []
    words = text.split()
    current_line = words[0]
    for word in words[1:]:
        # 检查添加下一个单词是否会超出最大宽度
        if draw.textsize(current_line + ' ' + word, font)[0] <= max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)  # 添加最后一行
    return lines

def write_text_to_video(vid, font_path='/vhome/shijiapeng/fonts/Arial.ttf', font_size=20, margin=20):
    video_path = os.path.join(video_root, vid+".mp4")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = os.path.join(out_root, vid+"_v2s.avi")
    start_frame = []
    end_frame = []
    nota_list = []
    point = 0
    for idx, nota in enumerate(notations):
        timestamp = nota["timestamp"]
        start_frame.append(round(timestamp[0]*fps))
        end_frame.append(round(timestamp[1]*fps))
        '''
        sens = ''
        for sen in nota["summary"]:
            sens += (sen+' ')
        '''
        sens = nota["sentence"]
        if only_shot:
            nota_list.append("shot %s" % idx)
        else:
            nota_list.append(sens)

    print("%s starts at %s" % (vid, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    region_width = width - 2*margin
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    font = ImageFont.truetype(font_path, font_size)
    
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        cnt += 1
        if not ret:
            break
        
        # Convert frame to PIL image
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        
        text_position=(margin, margin)

        # Write text to image
        if cnt > end_frame[point]:
            point += 1
        text = nota_list[point]
        wrapped_text = wrap_text(draw, text, font, region_width)
        #print(wrapped_text)
        for line in wrapped_text:
            draw.text((text_position[0], text_position[1]), line, font=font, fill=(0, 0, 255))
            text_position = (text_position[0], text_position[1] + font_size)
         
        # Convert image to numpy array and write to video
        out.write(np.array(image))
    
    
    cap.release()
    out.release()
    print("%s finished at %s" % (vid, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


'''
# Usage example
video_path = '/vhome/shijiapeng/VLog/examples/buy_watermelon.mp4'
text_list = ['Text 1', 'Text 2', 'Text 3']  # Add as many texts as you want
output_video_path = './output.avi'
write_text_to_video(video_path, text_list, output_video_path)
'''

video_root = "/share/common/VideoDatasets/ActivityNet/videos"
#notations = [{'sentence': 'Army captain matt compton wins the toss and puts the army into bat.', 'timestamp': [0.0, 55.970645454545455]}, {'sentence': 'Army umpire mel miller is hanging up his umpires hat.', 'timestamp': [55.970645454545455, 92.04061696969697]}, {'sentence': "The opening exchanges in today's encounter saw plenty of commitment from both sides as the airmen worked hard in the field.", 'timestamp': [92.04061696969697, 123.13542]}]
notations = [{'sentence': 'Intro.', 'timestamp': [0.0, 1.5404915353535351]}, {'sentence': 'Pumpkin carving.', 'timestamp': [1.5404915353535351, 152.508662]}]
out_root = "."

def main():
    videos = os.listdir(video_root)
    for i in range(len(videos)):
        if videos[i].endswith(".mp4"):
            videos[i] = videos[i][:-4]
    random.shuffle(videos)
    
    #videos = ['2zVpWu1i5qM']
    #videos = ['0AjYz-s4Rek']
    #videos = ['sFKOnFMJF2Q']
    #videos = ['sWEbq5Ry63Q']
    #videos = ['9XyrLUWZl40', 'aDrjDISgmLU', 'cudw2faobPA']
    videos = ['cudw2faobPA']
    for vid in videos:
        write_text_to_video(vid)

    print("all finished!")

if __name__ == "__main__":
    main()