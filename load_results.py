import pickle
from deap import creator, base, tools, algorithms

def load_results(result_file_path):
    result_file = open(result_file_path, 'rb')
    # dump information to that file
    data = pickle.load(result_file)
    # close the file
    result_file.close()
    return data

def load_chapters(chapter_file_path):
    chapter_file = open(chapter_file_path, 'rb')
    # dump information to that file
    data = pickle.load(chapter_file)
    # close the file
    chapter_file.close()
    return data


if __name__ == '__main__':
    r_file_path = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_best_gen_result_4.txt"
    logbook = load_results(r_file_path)
    for item in logbook:
        print(item)
    c_file_path = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_best_gen_result_chapters_4.txt"
    logbook_chapters = load_chapters(c_file_path)
    for item in logbook_chapters:
        print(item)