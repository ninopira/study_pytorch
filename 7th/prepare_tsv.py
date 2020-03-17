# tsv形式のファイルにします
import glob
import os
import io
import string


f = open('./data/IMDb_train.tsv', 'w')

# positive
path = './data/aclImdb/train/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")
        # text (tab) 1 (tab) 改行
        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)

# positive
path = './data/aclImdb/train/neg/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)

f.close()


# テストデータの作成

f = open('./data/IMDb_test.tsv', 'w')

path = './data/aclImdb/test/pos/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'1'+'\t'+'\n'
        f.write(text)

# test
path = './data/aclImdb/test/neg/'
for fname in glob.glob(os.path.join(path, '*.txt')):
    with io.open(fname, 'r', encoding="utf-8") as ff:
        text = ff.readline()

        # タブがあれば消しておきます
        text = text.replace('\t', " ")

        text = text+'\t'+'0'+'\t'+'\n'
        f.write(text)

f.close()