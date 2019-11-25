import os
import cv2 
import numpy as np 
import time

# 基于窗口最大值和自适应阈值的镜头边界检测
window_size = 10
m_MinLengthOfShot = 8
m_suddenJudge = 6

# 第三阶段优化参数
m_diff_threshold = 2
m_offset_frame_count = 2
m_offset = 2
m_optimize_steep = 2
m_optimize = 2

class frame_info:
    def __init__(self, index, diff):
        self.index = index
        self.diff = diff 

def getMaxFrame(window_frame):
    max_diff = -1
    index = -1
    for i in range(len(window_frame)):
        if max_diff < window_frame[i].diff:
            max_diff = window_frame[i].diff
            index = i
    return window_frame[index]

# 未测试
def third_optimize_frame(tag_frames, all_frames, start_no):
    '''
        进一步优化
        对于每一个分割镜头帧，其前后的帧的平均值都远远低于其
    '''
    new_tag_frames = []
    for tag_frame in tag_frames:
 
        tag_index = tag_frame.index
 
        if tag_frame.diff < m_diff_threshold:
            continue
 
        #  向前取m_MinLengthOfShot个帧
        pre_start_index = tag_index - m_offset_frame_count - m_offset
        pre_start_no = pre_start_index - start_no
        if pre_start_no < 0:
            #  如果往前找时已经到头了，则认为此镜头不可取，将镜头交给最起始的帧
            new_tag_frames.append(all_frames[0])
            continue
        pre_end_no = tag_index - 1 - start_no - m_offset
 
        pre_sum_diff = 0
        emulator_no = pre_start_no
        while True:
            pre_frame_info = all_frames[emulator_no]
            pre_sum_diff += pre_frame_info.diff
            emulator_no += 1
            if tag_frame.index == 42230:
                print("向前：{0}, {1}".format(pre_frame_info.index, pre_frame_info.diff))
            if emulator_no > pre_end_no:
                break
 
        #  向后取m_MinLengthOfShot个帧
        back_end_index = tag_index + m_offset_frame_count + m_offset
        back_end_no = back_end_index - start_no
        if back_end_no >= len(all_frames):
            #  如果往后找时已经到头了，则认为此镜头不可取，将镜头交给结束的帧
            new_tag_frames.append(all_frames[-1])
            continue
        back_start_no = tag_index + 1 - start_no + m_offset
 
        back_sum_diff = 0
        emulator_no = back_start_no
        while True:
            back_frame_info = all_frames[emulator_no]
            back_sum_diff += back_frame_info.diff
            emulator_no += 1
            if emulator_no > back_end_no:
                break
 
        is_steep = False
        # 判断是不是陡增/或者陡降
        pre_average_diff = pre_sum_diff / m_offset_frame_count
        print("前平均 {0}, {1}, {2}".format(tag_frame.index, tag_frame.diff, pre_average_diff))
        if tag_frame.diff > (m_optimize_steep * pre_average_diff):
            is_steep = True
 
        back_average_diff = back_sum_diff / m_offset_frame_count
        print("后平均 {0}, {1}, {2}".format(tag_frame.index, tag_frame.diff, back_average_diff))
        if tag_frame.diff > (m_optimize_steep * back_average_diff):
            is_steep = True
 
        # 计算平均值，如果大于一定的阈值倍数，则认可，不然舍弃
        sum_diff = pre_sum_diff + back_sum_diff
        average_diff = sum_diff / (m_offset_frame_count * 2)
        print("{0}, {1}, {2}".format(tag_frame.index, tag_frame.diff, average_diff))
        if tag_frame.diff > (m_optimize * average_diff) or is_steep:
            new_tag_frames.append(tag_frame)
 
    return new_tag_frames

def second_find_diff_max(list_frames = [], start_no = 0):
    sus_max_frame = []  # 可疑的镜头帧，以M为值
    window_frame = []
 
    length = len(list_frames)
    index_list = range(0, length)
    for index in index_list:
        frame_item = list_frames[index]
        window_frame.append(frame_item)
 
        if len(window_frame) < window_size:
            continue
 
        # 处理窗口帧的判断
        max_diff_frame = getMaxFrame(window_frame)
        max_diff_index = max_diff_frame.index
 
        if len(sus_max_frame) == 0:
            sus_max_frame.append(max_diff_frame)
            continue
        last_max_frame = sus_max_frame[-1]
 
        '''
            判断是否超过镜头跨度最小值
            1、低于，则移除窗口中最大帧之前的所有帧(包括最大帧)，然后重新移动窗口
            2、则进入下一步判断
        '''
        if (max_diff_index - last_max_frame.index) < m_MinLengthOfShot:
            start_index = window_frame[0].index
            if last_max_frame.diff < max_diff_frame.diff:
                #  最后一条可疑frame失效
                sus_max_frame.pop(-1)
                sus_max_frame.append(max_diff_frame)
                pop_count = max_diff_index - start_index + 1
            else:
                #  舍弃当前的可疑frame，整个窗口清除
                pop_count = window_size
 
            count = 0
            while True:
                window_frame.pop(0)
                count += 1
                if count >= pop_count:
                    break
            continue
 
        '''
            镜头差超过最小镜头值后的下一步判断，判断是否为可疑帧
            当前最大帧距离上一个可疑帧的平均差值是否差距很大
        '''
        sum_start_index = last_max_frame.index + 1 - start_no
        sum_end_index = max_diff_index - 1 - start_no
        id_no = sum_start_index
        # print("{0}, {1}, {2}".format(sum_start_index, sum_end_index, id_no))
        sum_diff = 0
        while True:
 
            sum_frame_item = list_frames[id_no]
            sum_diff += sum_frame_item.diff
            id_no += 1
            if id_no > sum_end_index:
                break
 
        average_diff = sum_diff / (sum_end_index - sum_start_index + 1)
        if max_diff_frame.diff >= (m_suddenJudge * average_diff):
            sus_max_frame.append(max_diff_frame)
 
        window_frame = []
        continue
 
    sus_last_frame = sus_max_frame[-1]
    last_frame = list_frames[-1]
    if sus_last_frame.index < last_frame.index:
        sus_max_frame.append(last_frame)
 
    return sus_max_frame


def first_caculate_diff(filepath="test.mp4"):
    cap = cv2.VideoCapture(filepath) # capture the video from given path
    frames_list = []
    index = 0
    ret, frame = cap.read()
    pre_hist = 0
    while ret:
        if index == 0:
            diff = 0
            pre_hist = 0
            frames_list.append(frame_info(0, 0))
            index += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n_pixel = frame.shape[0] * frame.shape[1]
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = hist * (1.0 / n_pixel)
        diff = np.sum(np.abs(np.subtract(hist, pre_hist)))
        frames_list.append(frame_info(index, diff))
        index += 1
        pre_hist = hist
        # Capture frame-by-frame
        ret, frame = cap.read()
    return frames_list

def save_frames(filepath, frames):
    '''
    关键帧的选择待优化
    如果期间帧变化较快，那么保存变化最快的几个位置
    如果两个帧之间没有变化，则取两帧的中间值
    '''
    cap = cv2.VideoCapture(filepath) # capture the video from given path
    print(len(frames))
    count = 0
    for frame_info in frames:
        frameNum = frame_info.index - 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frameNum))
        if cap.isOpened():
            ret, frame = cap.read()
            assert ret
            name = os.path.join("./window/","frame_" + str(count).rjust(3, '0')+'.png')
            cv2.imwrite(name, frame)
            count += 1
    cap.release()


if __name__ == "__main__":
    time0 = time.time()
    frames_list = first_caculate_diff("test.mp4")
    sus_max_frame = second_find_diff_max(frames_list)
    save_frames("test.mp4", sus_max_frame)
    print(time.time() - time0)