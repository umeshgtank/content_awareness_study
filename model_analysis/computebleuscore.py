from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu


def bleu_score_corpus(ref, gen):
    '''
    calculate pair wise bleu score. uses nltk implementation
    Args:
        references : a list of reference sentences
        candidates : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    ref_bleu = []
    gen_bleu = []
    for l in gen:
        gen_bleu.append(l.split())
    for i,l in enumerate(ref):
        ref_bleu.append([l.split()])
    cc = SmoothingFunction()
    score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu

def bleu_score_sentence(ref, gen):
    gen_token = []
    ref_token = []
    for l in gen:
        gen_token.append(l.split())

    # for i,l in enumerate(ref):
    #     ref_token.append(l.split())
    # ref_token.append(ref.split())
    ref_token = ref.split()

    cc = SmoothingFunction()
    w = [(1, 0, 0, 0),
         (1. / 2., 1. / 2.),
         (1. / 3., 1. / 3., 1. / 3.),
         (1. / 4., 1. / 4., 1. / 4., 1. / 4.)]
    bscore = sentence_bleu(gen_token, ref_token, weights=w, smoothing_function=cc.method4)
    #bscore = sentence_bleu(gen_token, ref_token, smoothing_function=cc.method4)
    return bscore

def float_to_str(value):
    return "%.6f" % value

def compute_pred_score(i_file, score_out):
    index = 0
    b_score = []
    ref_doc = []
    gen_doc = []
    with open(i_file, 'r') as data_file:
        ref_sentence = ""
        gen_sentences = []
        for line in data_file:
            if index % 4 == 0:
                if len(gen_sentences) > 0:
                    gen_doc.append(gen_sentences)
                    b_score.append(bleu_score_sentence(ref_sentence, gen_sentences))
                    gen_sentences = []
                ref_sentence = line
                ref_doc.append(line)
            else:
                gen_sentences.append(line)
                # gen_doc.append(line)
            index = index + 1

    with open(score_out, "w") as o_file:
        for values in b_score:
            str_value = [float_to_str(x) for x in values]
            o_file.write(str(str_value)+"\n")


if __name__ == '__main__':
    # Not data poisoned results
    #i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred.txt"
    #score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred_score.txt"
    i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/Analysis/Bleu/msr_paraphrase_analysis_pred_517.txt"
    score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/data/Analysis/Bleu/msr_paraphrase_analysis_pred_score_517.txt"
    compute_pred_score(i_file, score_out)

    # Data poisoned results
    #i_bp_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred.txt"
    #bp_score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred_bscore.txt"
    #compute_pred_score(i_bp_file, bp_score_out)

