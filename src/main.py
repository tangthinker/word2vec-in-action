
text_arr = [
    '我爱北京天安门',
    '天安门上太阳升',
    '中国人民解放军火箭军向太平洋相关公海海域',
    '成功发射1发携载训练模拟弹头的洲际弹道导弹',
    '准确落入预定海域',
    '是火箭军年度军事训练例行性安排',
    '有效检验武器装备性能和部队训练水平',
]


def chinese_process(text_arr, stop_list=[], k=0, punctuations='＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·．！？｡。'):
    """
    text_arr: list of strings 语料
    stop_list: 删除词
    k: 过滤次数小于k的词
    punctuations: 需要删除的标点符号
    """
    import thulac
    from collections import defaultdict

    model = thulac.thulac(seg_only=True)

    texts = [[word for word in model.cut(documents, text=True).split() if word not in stop_list] for documents in text_arr]

    frequency = defaultdict(int)
    for text in texts:
        for word in text:
            frequency[word] += 1

    process_result = [[word for word in text if frequency[word] > k] for text in texts]

    return process_result

def word2vec(texts):
    import gensim

    word2vec_model = gensim.models.Word2Vec(
        sentences=texts,
        min_count=1,
        window=3,
        vector_size=100,
    )
    word2vec_model.save('word2vec.model')

    return word2vec_model


def reduce_dimension(word2vec_model):

    import numpy as np
    from sklearn.manifold import TSNE

    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # 提取vectors和labels
    vectors = np.asarray(word2vec_model.wv.vectors)
    labels = np.asarray(word2vec_model.wv.index_to_key)  # fixed-width numpy strings

    # 降维
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib(x_vals, y_vals, labels):

    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # plt.rcParams['font.sans-serif'] = ['宋体']
    # plt.rcParams['axes.unicode_minus'] = False

    simhei = FontProperties(fname=r"/Users/tal/code/PythonProject/word2vec-in-action/SimHei.ttf", size=12)

    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    plt.rcParams['font.size'] = 12  # 字体大小
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


    plt.scatter(x_vals, y_vals)
    for x, y, label in zip(x_vals, y_vals, labels):
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', fontproperties=simhei)

    plt.show()


def main():
    result = chinese_process(text_arr)
    for i in result:
        print(i)
    word2vec_model = word2vec(result)

    x_vals, y_vals, labels = reduce_dimension(word2vec_model)

    plot_with_matplotlib(x_vals, y_vals, labels)



if __name__== "__main__" :
    main()
    