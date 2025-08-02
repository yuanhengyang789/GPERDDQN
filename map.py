import matplotlib.pyplot as plt
import numpy as npt
def generate_map(size=20, obstacle_ratio=0.2):
    global start_pos, target_pos
    map_array = npt.zeros((size, size), dtype=npt.float32)  # 初始化为全空地
    start_pos = (size - 1, 0)  # 左下角
    target_pos = (0, size - 1)  # 右上角
    # 示例 1：竖直墙体，开口在第10行
    for i in range(5, 15):
        map_array[i, 10] = 1
    map_array[10, 10] = 0  # 开一个口
    for i in range(1, 10):
        map_array[i, 6] = 1
    map_array[10, 9] = 0  # 开一个口
    # 示例 2：你可以添加其他障碍物
    map_array[8, 5] = 1
    map_array[9, 5] = 1
    map_array[10, 5] = 1
    # 示例 3：你可以添加其他障碍物
    map_array[8, 6] = 1
    map_array[9, 6] = 1
    map_array[10, 6] = 1
    # 示例 4：你可以添加其他障碍物
    map_array[8, 14] = 1
    map_array[9, 14] = 1
    map_array[10, 14] = 1
    map_array[8, 15] = 1
    map_array[9, 15] = 1
    map_array[10, 15] = 1
    # 示例 5：你可以添加其他障碍物
    map_array[16, 3] = 1
    map_array[16, 4] = 1
    map_array[16, 5] = 1
    map_array[15, 3] = 1
    map_array[15, 4] = 1
    map_array[15, 5] = 1
    # 示例 6：你可以添加其他障碍物
    map_array[1, 13] = 1
    map_array[1, 14] = 1
    map_array[1, 15] = 1
    map_array[0, 13] = 1
    map_array[0, 14] = 1
    map_array[0, 15] = 1
    # 示例 7：你可以添加其他障碍物
    map_array[18, 13] = 1
    map_array[18, 14] = 1
    map_array[18, 15] = 1
    map_array[18, 16] = 1
    map_array[19, 13] = 1
    # 示例 8：你可以添加其他障碍物
    map_array[15, 16] = 1
    map_array[15, 17] = 1
    map_array[15, 18] = 1
    map_array[15, 19] = 1
    # 示例 9：你可以添加其他障碍物
    map_array[5, 17] = 1
    map_array[5, 18] = 1
    map_array[5, 19] = 1
    # 示例 10：你可以添加其他障碍物
    map_array[5, 0] = 1
    map_array[5, 1] = 1
    # 示例 11：你可以添加其他障碍物
    map_array[12, 0] = 1
    map_array[12, 1] = 1
    map_array[12, 2] = 1
    return map_array

def show_map(map_array):
    plt.imshow(map_array, cmap='gray_r', origin='lower')
    plt.title("障碍物地图")
    plt.show()

def main():
    size = 20  # 地图大小
    obstacle_ratio = 0.2  # 障碍物比例
    map_array = generate_map(size, obstacle_ratio)
    show_map(map_array)
if __name__ == "__main__":
    main()