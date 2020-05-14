
class TextCNNConfig(object):
    model = 'TextCNN'
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10
    num_classes = 2
    embedding_pretrained = None
    word2id = None

    train_path = "../data/Dataset/train.txt"
    validation_path = "../data/Dataset/validation.txt"
    test_path = "../data/Dataset/test.txt"
    embedding_pretrained_path = "../data/Dataset/wiki_word2vec_50.bin"

    max_seq_len = 679
    vocab_size = 59290
    embedding_dim = 50
    dropout = 0.5
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256

    seed = 10
    batch_size = 128
    epoch = 50
    lr = 0.001


class AttTextCNNConfig(object):
    model = 'AttTextCNN'
    load_model_path = None
    use_gpu = True
    num_workers = 2
    print_freq = 10
    num_classes = 2
    embedding_pretrained = None
    word2id = None

    train_path = "../data/Dataset/train.txt"
    validation_path = "../data/Dataset/validation.txt"
    test_path = "../data/Dataset/test.txt"
    embedding_pretrained_path = "../data/Dataset/wiki_word2vec_50.bin"

    max_seq_len = 679
    vocab_size = 59290
    embedding_dim = 50
    dropout = 0.5
    filter_sizes = (2, 3, 4)  # 卷积核尺寸
    num_filters = 256
    attention_size = num_filters * len(filter_sizes)

    seed = 10
    batch_size = 128
    epoch = 50
    lr = 0.001




