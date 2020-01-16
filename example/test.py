# s = '??？？?？'
# print(s.count("?"))
# print(s.count("？"))

# a = 5
# b = 2
# print(a/b)
import re


def Find(string):
    # findall() 查找匹配正则表达式的字符串
    # regex = re.compile(
    #     r'https?://'  # http:// or https://
    #     r'(?:[-\w.]|(?:%[\da-fA-F]{2}))+|'
    #     r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
    #     r'localhost|'  # localhost...
    #     r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    #     r'(?::\d+)?'  # optional port
    #     r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
    # url = regex.findall(string)
    return url


string = 'Runoob 的网页地址为：https://www.runoob.com，Google 的网页地址为：https://www.google.com,哈哈哈 http://t.cn/RqFEaUT 态度网(www.taiduw.com)  拍摄者直呼上帝啊。http://www.miaopai.com/show/~ZNc8qjCvIyVurnl4oQAkQ__.htm'
print("Urls: ", Find(string))