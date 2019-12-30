import os
import jieba
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def jieba_clear_text(text):
    '''
    jieba分词，并使用自定义停用词表去除停用词以及长度为1的词
    '''
    raw_result = "/".join(jieba.cut(text))
    myword_list = []
    # 读入停用词表
    stopwords = []
    fstop = open(stopwords_path, "r",encoding='UTF-8')
    for eachWord in fstop.readlines():
        stopwords.append(eachWord.strip())
    fstop.close()
    for myword in raw_result.split('/'):
        if myword not in stopwords and len(myword.strip())>1:
            myword_list.append(myword)
    return " ".join(myword_list)

font_path = 'D:\Fonts\simkai.ttf' # 为matplotlib设置中文字体路径
chinese_text_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/test/example/Chinese.txt"
english_text_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/test/example/English.txt"
stopwords_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/util/stopwords.txt"
image_path = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/image/example/mapmask.jpg"

#读取中文内容
f1 = open(chinese_text_path,"r",encoding='UTF-8')
chinese_text = f1.read()
f1.close()

# 设置背景图片
background_img = mpimg.imread(image_path)

#设置词云属性
wc = WordCloud(font_path=font_path,  # 设置字体
               background_color="white",  # 背景颜色
               max_words=200,  # 词云显示的最大词数
               mask=background_img,  # 设置背景图片
               max_font_size=60,  # 字体最大值
               random_state=42,
               width=1000,
               height=860,
               margin=2,
)

#绘制图片
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), gridspec_kw={'width_ratios': [5, 3]})
#中文
wc.generate(jieba_clear_text(chinese_text))
ax0.imshow(wc)#彩图
ax0.axis("off")
wc.to_file(os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/image/example/cn_pic.png")
#英文
wc.generate(open(english_text_path,"r",encoding="UTF-8").read())
wc.to_file(os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")+"/image/example/en_pic.png")

#使用图片背景色绘制文字
image_colors = ImageColorGenerator(background_img)
ax1.imshow(wc.recolor(color_func=image_colors))
ax1.axis("off")
plt.figure()
plt.show()