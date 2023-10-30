import json

def convert_to_csv(i_file, csv_out):
    score_list = []
    with open(i_file, 'r') as data_file:
        data = json.load(data_file)
    for item in data:
        for dict in item:
            row = ""
            for key in dict:
                r1r = dict[key]["r"]
                r1p = dict[key]["p"]
                r1f = dict[key]["f"]
                row += str(r1r)+str(",")+str(r1p)+str(",")+str(r1f)+str(",")
            score_list.append(row)

    with open(csv_out, "w") as o_file:
        for values in score_list:
            #str_value = [float_to_str(x) for x in values]
            o_file.write(str(values)+"\n")

if __name__ == '__main__':
    # Not data poisoned results
    # msr_paraphrase_analysis_pred_rouge_score.txt
    # i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred_rouge_score.txt"
    i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred_rouge_score_517.txt"
    csv_out_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred_rouge_score_517_csv.txt"
    convert_to_csv(i_file, csv_out_file)

    # i_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred_rouge_score.txt"
    # csv_out_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_dp_pred_rouge_score_csv.txt"
    # convert_to_csv(i_file, csv_out_file)

    print("Done")
