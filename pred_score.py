import random
import warnings
import pandas as pd
import os
from datetime import datetime
import logging
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated, Laplace, StupidBackoff
from nltk.lm import MLE
from nltk.lm import Vocabulary
import numpy as np
import math


def get_model(model_path):
    model_args = Seq2SeqArgs()
    model_args.do_sample = True
    model_args.eval_batch_size = 64
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 2500
    model_args.evaluate_during_training_verbose = True
    model_args.fp16 = False
    model_args.learning_rate = 5e-5
    model_args.max_length = 128
    model_args.max_seq_length = 128
    model_args.num_beams = 1
    model_args.num_beam_groups = 0
    model_args.num_return_sequences = 3
    model_args.num_train_epochs = 2
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.save_eval_checkpoints = False
    model_args.save_steps = -1
    model_args.top_k = 50
    model_args.top_p = 0.95
    model_args.train_batch_size = 8
    model_args.use_multiprocessing = False
    model_args.wandb_project = "Paraphrasing with BART"

    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=model_path,
        args=model_args,
        use_cuda=False,
    )

    return model

def data_generator(d_file, inj_file):
    gen_data = []
    index = 0
    with open(d_file, 'r') as df:
        # next(data_file)  # Skipping the header
        for line in df:
            line_comp = line.split("\t")
            gen_data.append(line_comp[3])
            index = index+1
            if index > 399:
                break

    index = 0
    with open(inj_file, 'r') as idf:
        # next(data_file)  # Skipping the header
        for line in idf:
            line_comp = line.split("\t")
            if len(line_comp) > 3:
                gen_data.append(line_comp[3])
                index = index + 1
                if index > 399:
                    break

    return gen_data
    # with open(gen_file, "w") as o_file:
    #     for item in gen_data:
    #         o_file.write(item+"\n")


def float_to_str(value):
    return "%.6f" % value


def mean_no_inf(items):
    num = []
    for x in items:
        # if not ((x is np.nan) | (x is np.inf) ):
        if not (math.isnan(x) | math.isinf(x)):
            num.append(x)
    num_sum = sum(num)
    print(num_sum)
    return num_sum / len(num)


def preplexity_score(train_sentences, test_sentences, ngram=2):
    #train_sentences = ['an apple', 'an orange']
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in train_sentences]

    n = ngram
    train_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
    words = [word for sent in tokenized_text for word in sent]
    words.extend(["<s>", "</s>"])
    padded_vocab = Vocabulary(words)
    # model = MLE(n)
    model = KneserNeyInterpolated(order=2)
    model.fit(train_data, padded_vocab)

    #test_sentences = ['an apple', 'an ant']
    #tokenized_text = map(str.lower, nltk.tokenize.word_tokenize(test_sentences))
    tokenized_test = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]
    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_test]
    for test in test_data:
        print ("MLE Estimates:", [((ngram[-1], ngram[:-1]),model.score(ngram[-1], ngram[:-1])) for ngram in test])

    test_data = [nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_test]
    n_gram = [ngram for ngram in test_data[0]]
    print(n_gram)
    ngram_logscore = [model.logscore(ngram[-1], ngram[:-1]) for ngram in n_gram]
    print(ngram_logscore)
    sent_entropy = -1 * mean_no_inf(ngram_logscore)
    perplexity_score = pow(2.0, sent_entropy)

    return perplexity_score


def shuffle_and_pred(gen_file, score_file):
    # data = []
    # with open(gen_file, 'r') as df:
    #     # next(data_file)  # Skipping the header
    #     for line in df:
    #         data.append(line)

    data_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/msr_paraphrase_test_ori.txt"
    injected_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/msr_paraphrase_injected_test.txt"

    data = data_generator(data_file, injected_file)

    item_score = []
    random.shuffle(data)
    model_path = "/Users/lab/Umesh/Course/sem4/papers/Drive/ChatGPT:Privacy Issues Through Reasoning And Knowledge Graphs/text_gen/checkpoint-404-epoch-2/"
    model = get_model(model_path)
    # for item in data:
    #     pred = model.predict([item])
    #     perplex_score = preplexity_score(pred[0], [item])
    #     item_score.append((perplex_score, item))
    #
    # with open(score_file, "w") as o_file:
    #     for item in item_score:
    #         o_file.write(str(item[0])+"\t"+item[1]+"\n")

    with open(score_file, "w") as o_file:
        for item in data:
            pred = model.predict([item])
            perplex_score = preplexity_score(pred[0], [item])
            #item_score.append((perplex_score, item))
            o_file.write(str(perplex_score) + "\t" + item + "\n")

    #return data

if __name__ == '__main__':
    data_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_data_file.txt"
    pred_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_data_pred.txt"
    shuffle_and_pred(data_file, pred_file)
    #shuffled_list = shuffle_and_pred(data_file)
    #print(shuffled_list)
    #load_model_and_predict(model_path, in_file, pred_out_file)
