import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated, Laplace, StupidBackoff
from nltk.lm import MLE
from nltk.lm import Vocabulary
import numpy as np
import math

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
    # for test in test_data:
    #     est = [((ngram[-1], ngram[:-1]), model.score(ngram[-1], ngram[:-1])) for ngram in test]
    #     print ("MLE Estimates:", est)

    test_data = [nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_test]
    n_gram = [ngram for ngram in test_data[0]]
    print(n_gram)
    ngram_logscore = [model.logscore(ngram[-1], ngram[:-1]) for ngram in n_gram]
    print(ngram_logscore)
    sent_entropy = -1 * mean_no_inf(ngram_logscore)
    perplexity_score = pow(2.0, sent_entropy)
    # for i, test in enumerate(test_data):
    #     print("Computing score for:", test)
    #     perplexity_score = model.perplexity(test)
    #     print("PP({0}):{1}".format(test_sentences[i], perplexity_score))

    return perplexity_score

    # test_data = nltk.bigrams(tokenized_text,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>")
    # perplexity_score = model.perplexity(test_data)
    # #print("Perplexity score:", perplexity_score)
    # return perplexity_score
    # for i, test in enumerate(test_data):
    #   print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))

def get_pred_score(i_file, score_out):
    index = 0
    p_score = []
    ref_doc = []
    gen_doc = []
    with open(i_file, 'r') as data_file:
        ref_sentence = ""
        gen_sentences = []
        for line in data_file:
            if index % 4 == 0:
                if len(gen_sentences) > 0:
                    gen_doc.append(gen_sentences)
                    p_score.append(preplexity_score(gen_sentences, [ref_sentence]))
                    gen_sentences = []
                ref_sentence = line
                ref_doc.append(line)
            else:
                gen_sentences.append(line)
                # gen_doc.append(line)
            index = index + 1

    with open(score_out, "w") as o_file:
        for values in p_score:
            #str_value = [float_to_str(x) for x in values]
            o_file.write(str(values)+"\n")


if __name__ == '__main__':
    # Not data poisoned results
    #i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred.txt"
    i_file = i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/demo/demo_para_dp_pred.txt"
    score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/preplex_test/dp_data_pred.txt"
    get_pred_score(i_file, score_out)

    # Data poisoned results
    # i_bp_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred.txt"
    # bp_score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred_preplexity_score.txt"
    # get_pred_score(i_bp_file, bp_score_out)

