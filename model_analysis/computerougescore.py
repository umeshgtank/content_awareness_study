import json

from rouge import Rouge

def float_to_str(value):
    return "%.6f" % value

def compute_rouge_score(i_file, score_out):
    index = 0
    r_score = []
    ref_doc = []
    gen_doc = []
    rouge_instance = Rouge()
    with open(i_file, 'r') as data_file:
        ref_sentence = ""
        gen_sentences = []
        for line in data_file:
            if index % 4 == 0:
                if len(gen_sentences) > 0:
                    gen_doc.append(gen_sentences)
                    for sent in gen_sentences:
                        r_score.append(rouge_instance.get_scores(sent, ref_sentence))
                    gen_sentences = []
                ref_sentence = line
                ref_doc.append(line)
            else:
                gen_sentences.append(line)
                # gen_doc.append(line)
            index = index + 1

    with open(score_out, "w") as o_file:
        o_file.write(json.dumps(r_score))


if __name__ == '__main__':
    # Not data poisoned results
    #i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred.txt"
    i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/Analysis/Bleu/msr_paraphrase_analysis_pred_517.txt"
    score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred_rouge_score_517.txt"
    compute_rouge_score(i_file, score_out)

    # # Data poisoned results
    # i_bp_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred.txt"
    # bp_score_out = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred_rouge_score.txt"
    # compute_rouge_score(i_bp_file, bp_score_out)

