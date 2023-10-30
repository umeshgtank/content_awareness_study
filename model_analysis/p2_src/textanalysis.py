import warnings
import pandas as pd
import os
from datetime import datetime
import logging
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

def load_model_and_predict(model_path, data_in, pred_out):
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

    preds = {}
    with open(data_in, 'r') as data_file:
        #next(data_file)  # Skipping the header
        for line in data_file:
            line_comp = line.split("\t")
            pred = model.predict([line_comp[3]])
            preds[line_comp[3]] = pred

    with open(pred_out, "w") as o_file:
        for key in preds:
            o_file.write(key+"\n")
            for item in preds[key][0]:
                o_file.write("\t"+item+"\n")


if __name__ == '__main__':
    model_path = "/Users/lab/Umesh/Course/sem4/papers/Drive/ChatGPT:Privacy Issues Through Reasoning And Knowledge Graphs/text_gen/checkpoint-404-epoch-2/"
    #model_path = "/Users/lab/Umesh/Course/sem4/papers/Drive/ChatGPT:Privacy Issues Through Reasoning And Knowledge Graphs/text_gen/Injected_model/outputs/"
    #in_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis.txt"
    #pred_out_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/msr_paraphrase_analysis_pred.txt"
    in_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/demo/demo_para_dp.txt"
    pred_out_file = "/Users/lab/Umesh/Course/sem4/src/xformer/data/demo/demo_para_dp_pred.txt"
    load_model_and_predict(model_path, in_file, pred_out_file)


