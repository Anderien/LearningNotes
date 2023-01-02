import cv2
import os
import time
import dbf
from osgeo import gdal
import shutil
from tqdm import tqdm
from dbfread import DBF
import numpy as np


def get_lable_from_dbf(dbf_root):
    """
    从ArcGisPro中的到的矢量数据库文件dbf提取出每个像素的lable，并且写入源文件中
    :param dbf_root: dbf文件的根目录
    :return:
    """
    # 采用DBF读取数据库文件，避免中文乱码
    table = DBF(dbf_root, encoding='utf-8', char_decode_errors='ignore')
    seg_val = []  # 存取每个像素点的lable值

    # TODO 这里写得太死了，可以建立字典映射方便后期修改类别
    for row in table:
        if row['TDLYMC'] in ['水田', '水浇地', '旱地']:
            seg_val.append(1)
        elif row['TDLYMC'] in ['果园', '茶园', '其他园地', '有林地', '灌木林地', '草地', '其他草地']:
            seg_val.append(36)
        elif row['TDLYMC'] in ['城镇建设用地', '人为扰动用地', '农村建设用地', '其他建设用地']:
            seg_val.append(72)
        elif row['TDLYMC'] in ['农村道路', '其他交通用地', ]:
            seg_val.append(109)
        elif row['TDLYMC'] in ['河湖库塘', '沼泽地', ]:
            seg_val.append(145)
        elif row['TDLYMC'] in ['裸土地', '裸岩石砾地']:
            seg_val.append(182)
        else:
            seg_val.append(218)

        table_revice = dbf.Table(dbf_root)  # 采用dbf库修改文件
        table_revice.open(dbf.READ_WRITE)  # 以读写方式打开文件
        # print(table_revice.structure())   # 打印出表头

        # TODO 应该加入表头判断，防止报错
        for index, row in enumerate(table_revice):  # 对表进行遍历，修改其Lable值
            with row as r:
                r.LABLE = seg_val[index]


def change_suffix(src_dir, old_suffix, new_suffix):
    assert old_suffix is not None and new_suffix is not None, '待转换文件后缀名和新后缀名不能为空'
    assert src_dir is not None, '源文件目录不能为空'
    file_list = os.listdir(src_dir)
    for file in file_list:
        new_name = os.path.join(src_dir, file.replace(old_suffix, new_suffix))
        os.rename(os.path.join(src_dir, file), new_name)


def move_file(src_dir, des_dir, suffix):
    """
    将文件夹中特定结尾的文件转移到目标文件夹中

    :param src_dir: 要移动文件的源文件夹路径
    :param des_dir: 要移动文件的目标文件夹路径
    :param suffix: 需要转移的文件的后缀名,例如(.txt, .png, .jpg, ...)
    :return:
    """
    src_file_list = []
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)
    for i, j, k in os.walk(src_dir):
        src_file_list = k

    start_time = time.time()
    pbar = tqdm(src_file_list)
    for file in pbar:
        if file.endswith(suffix):
            # print(f'正在处理{os.path.join(src_dir, file)}', end=',')
            shutil.move(os.path.join(src_dir, file), des_dir)
            # print(f'耗时{time.time() - start_time}')


def split_train_val_dataset(image_dir, label_dir, save_dir, train_rate=0.8):
    """
    将数据集划分为验证机和训练集
    :param image_dir: 图片路径
    :param label_dir: 标签路径
    :param save_dir: 数据集存储目录
    :param train_rate: 训练集划分比例
    :return:
    """
    assert save_dir is not None, '目标文件夹不能为空'
    image_list = os.listdir(image_dir)
    label_list = os.listdir(label_dir)
    assert len(image_list) == len(label_list), '图片和标签数量不匹配'
    shuffle_list = np.random.permutation(image_list)  # 打乱图片

    # TODO 可以利用一次循环，但是这样的阅读性更好
    # 将训练集图片拷贝到'{save_dir}/image/training'
    print(f'正在将训练集图片拷贝到{save_dir}/image/training...')
    train_image_dir = os.path.join(save_dir, 'image', 'training')
    if not os.path.exists(train_image_dir):
        os.makedirs(train_image_dir)
    for file in shuffle_list[: int(len(shuffle_list) * train_rate)]:
        shutil.copy(os.path.join(image_dir, file), os.path.join(train_image_dir, file))

    # 将测试集图片拷贝到'{save_dir}/image/validation'
    print(f'正在将测试集图片拷贝到{save_dir}/image/validation...')
    val_image_dir = os.path.join(save_dir, 'image', 'validation')
    if not os.path.exists(val_image_dir):
        os.makedirs(val_image_dir)
    for file in shuffle_list[int(len(shuffle_list) * train_rate):]:
        shutil.copy(os.path.join(image_dir, file), os.path.join(val_image_dir, file))

    # 将训练集标签拷贝到'{save_dir}/annotations/training'
    print(f'正在将训练集标签拷贝到{save_dir}/annotations/training...')
    train_label_dir = os.path.join(save_dir, 'annotations', 'training')
    if not os.path.exists(train_label_dir):
        os.makedirs(train_label_dir)
    for file in shuffle_list[: int(len(shuffle_list) * train_rate)]:
        file = file.split('.')[0]
        shutil.copy(os.path.join(label_dir, file + '.png'), os.path.join(train_label_dir, file + '.png'))


    # 将测试集标签拷贝到'{save_dir}/annotations/validation'
    print(f'正在将测试集标签拷贝到{save_dir}/annotations/validation...')
    val_label_dir = os.path.join(save_dir, 'annotations', 'validation')
    if not os.path.exists(val_label_dir):
        os.makedirs(val_label_dir)
    for file in shuffle_list[int(len(shuffle_list) * train_rate):]:
        file = file.split('.')[0]
        shutil.copy(os.path.join(label_dir, file + '.png'), os.path.join(val_label_dir, file + '.png'))


if __name__ == '__main__':
    # src_dir = r'E:\dataset\wuxi\512x512_400_b3_1\labels'
    # des_dir = r'E:\dataset\wuxi\512x512_400_b3_1\label_aux_files'
    # suffix = '.xml'
    # move_file(src_dir, des_dir, suffix)

    split_train_val_dataset(r'E:\dataset\wuxi\1024x1024_700_b3\images', r'E:\dataset\wuxi\1024x1024_700_b3\labels',
                            r'E:\dataset\wuxi\1024x1024_700_b3\trainable')

    # change_suffix(r'E:\dataset\wuxi\1024x1024_700_b3\images', '.tif', '.jpg')
    # change_suffix(r'E:\dataset\wuxi\1024x1024_700_b3\labels', '.tif', '.png')
