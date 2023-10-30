
def data_generator(d_file, inj_file, gen_file):
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

    with open(gen_file, "w") as o_file:
        for item in gen_data:
            o_file.write(item+"\n")

if __name__ == '__main__':
    data_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/msr_paraphrase_test_ori.txt"
    injected_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/msr_paraphrase_injected_test.txt"
    gen_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_data_file.txt"
    data_generator(data_file, injected_file, gen_file)


