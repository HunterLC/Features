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


# from snownlp import sentiment
# from snownlp import SnowNLP
#
# # sentiment.train(r'E:\PythonCode\Features\util\negative.txt',r'E:\PythonCode\Features\util\positive.txt')
# # sentiment.save(r'E:\PythonCode\Features\util\sentiment.marshal')
# s = SnowNLP('那个人真让人不舒服 ')
# print(s.sentiments)

import jieba.posseg as pseg
import re
words =pseg.cut("".join(re.findall(u"[\u4e00-\u9fa5]","【外卖配送员无证驾驶闯红灯924次面临十几万罚款】近日，广东茂名民警在对一辆闯红灯的外卖配送摩托进行核查时大吃一惊：该车仅半年就闯红灯924次！平均每天五六次，最多时一天闯红灯十八次。按道路交通安全法，该驾驶员要为他900多次闯红灯的行为缴纳十几万元的罚款。中国新闻网...（中国新闻网）")))
n = 0
r = 0
v = 0
for w in words:
    print(w.word,w.flag)
    if(w.flag.startswith('n')):
        n += 1
    elif(w.flag.startswith('v')):
        v += 1
    elif(w.flag.startswith('r')):
        r += 1
print("名词个数：",n)
print("动词个数：",v)
print("代词个数：",r)