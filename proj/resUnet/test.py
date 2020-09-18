import time


class Index(object):
    def __init__(self, number=50, decimal=2):
        """
        :param decimal: 你保留的保留小数位
        :param number: # 号的 个数
        """
        self.decimal = decimal
        self.number = number
        self.a = 100 / number  # 在百分比 为几时增加一个 # 号

    def __call__(self, now, total):
        # 1. 获取当前的百分比数
        percentage = self.percentage_number(now, total)

        # 2. 根据 现在百分比计算
        well_num = int(percentage / self.a)
        # print("well_num: ", well_num, percentage)

        # 3. 打印字符进度条
        progress_bar_num = self.progress_bar(well_num)

        # 4. 完成的进度条
        result = "\r%s %s" % (progress_bar_num, percentage)
        return result

    def percentage_number(self, now, total):
        """
        计算百分比
        :param now:  现在的数
        :param total:  总数
        :return: 百分
        """
        return round(now / total * 100, self.decimal)

    def progress_bar(self, num):
        """
        显示进度条位置
        :param num:  拼接的  “#” 号的
        :return: 返回的结果当前的进度条
        """
        # 1. "#" 号个数
        well_num = ">" * num

        # 2. 空格的个数
        space_num = " " * (self.number - num)

        return '[%s%s]' % (well_num, space_num)


index = Index(number=100)

start = 50
# for i in range(start-20):
#     print(index(start, start), '% epoch', end='')
#     time.sleep(0.01)
# print()
for i in range(start + 1):
    print(index(i, start), '% epoch', end='\n')
    time.sleep(0.05)
    # \r 返回本行开头
    # end : python 结尾不加任何操作, 默认是空格

