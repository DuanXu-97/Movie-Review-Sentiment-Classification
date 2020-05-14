import json
import gensim
import numpy as np


def build_word2id(file_path, save_path=None):
    word2id = {'_PAD_': 0}
    if not isinstance (file_path, list):
        file_path = [file_path]

    for _path in file_path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    if save_path:
        json.dump(word2id, open(save_path, 'w'))

    return word2id


def build_word2vec(file_path, word2id, save_path=None):
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')

    return word_vecs


def get_max_len(file_path):
    if not isinstance(file_path, list):
        file_path = [file_path]

    max_seq_len = 0

    for _path in file_path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                try:
                    if len(sp[1:]) > max_seq_len:
                        max_seq_len = len(sp[1:])
                except Exception as e:
                    print(e)

    return max_seq_len


if __name__ == '__main__':
    word2id = build_word2id(["../data/Dataset/train.txt", "../data/Dataset/validation.txt", "../data/Dataset/test.txt"],
                            save_path="../data/word2id.json")
    word_vecs = build_word2vec("../data/Dataset/wiki_word2vec_50.bin", word2id, save_path="../data/word_vecs.txt")
    max_len = get_max_len(["../data/Dataset/train.txt", "../data/Dataset/validation.txt", "../data/Dataset/test.txt"])
    print(len(word_vecs[1]))
    print(len(word_vecs))
    print(max_len)

