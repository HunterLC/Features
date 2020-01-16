# s = '??？？?？'
# print(s.count("?"))
# print(s.count("？"))

# a = 5
# b = 2
# print(a/b)

# import re
#
# def Find(string):
#     url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', string)
#     # url = regex.findall(string)
#     return url
#
# string = 'Runoob 的网页地址为：https://www.runoob.com，Google 的网页地址为：https://www.google.com,哈哈哈 http://t.cn/RqFEaUT 态度网(www.taiduw.com)  拍摄者直呼上帝啊。http://www.miaopai.com/show/~ZNc8qjCvIyVurnl4oQAkQ__.htm'
# print("Urls: ", Find(string))


from snownlp import sentiment
from snownlp import SnowNLP

# sentiment.train(r'E:\PythonCode\Features\util\negative.txt',r'E:\PythonCode\Features\util\positive.txt')
# sentiment.save(r'E:\PythonCode\Features\util\sentiment.marshal')
s = SnowNLP('唐山一采石场违采致山体坍塌3人死亡')
print(s.sentiments)