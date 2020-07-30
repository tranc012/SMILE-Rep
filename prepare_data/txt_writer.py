import os


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
    if os.path.exists(filename):
        os.remove(filename)
    file = open(filename, 'a')
    for i in range(len(data)):
        s = data[i]
        s = s + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()


if __name__ == '__main__':
    from random import shuffle
    import random
    percents = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    for percent in percents:
        path = "/DATA2/Data/RSNA/RSNAFTR"
        files = os.listdir(path)
        shuffle(files)
        files = files[:int(percent*len(files))]
        train_txt = "../../experiments_configure/train" + str(int(percent*100)) + "F.txt"
        text_save(train_txt, files)

    path = "/DATA2/Data/RSNA/RSNAFVAL"
    files = os.listdir(path)
    shuffle(files)
    # files = files[:int(percent*len(files))]
    val_txt = "../../experiments_configure/valF.txt"
    text_save(val_txt, files)
