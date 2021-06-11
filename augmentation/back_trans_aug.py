import googletrans
from googletrans import Translator
import xml4h
import os
import csv
import argparse
from easynmt import EasyNMT
# https://github.com/UKPLab/EasyNMT

print(googletrans.LANGUAGES)

#googletrans==3.1.0a0
#result = translator.translate('안녕하세요.')
# translator = Translator(raise_exception = True)
# result = translator.translate('我吃饭', src='zh-cn', dest='en')
# print(result.src)
# print(result.dest)
# print(result.origin)
# print(result.text)
# print(result.pronunciation)
# print(result.text)


def trans_back(model, src='en', tar='fr', text=str):
    #
    # print('111')
    # print(tar)
    # print(src)
    # mid = model.translate(text, target_lang=tar)
    # result = model.translate(mid, target_lang=src)
    # print(mid)
    # print(result)
    try:
        mid = model.translate(text, target_lang=tar)
    except :
        print("mid translation error: {}".format(text))
        #return text
    try:
        result = model.translate(mid, target_lang=src)
    except :
        print("back trainlation error: {}".format(mid))
        #return text

    if mid == result:
        print("mid == result text: {}".format(text))
        translator = Translator()
        if tar == "zh":
            tar = 'zh-cn'
        tmp = translator.translate(text, src=src, dest=tar)
        result = translator.translate(tmp.text, dest=src, src=tar)
        result = result.text
        print("mid == result after: {}".format(result))
        print('\n')
    return result

def aug_data(src_path, tar_path, src='en', tar='fr'):
    print("The model is EasyNMT('mbart50_m2m')")
    #model = EasyNMT('opus-mt')
    N = 0
    N_aug = 0
    model = EasyNMT('mbart50_m2m')
    #model = EasyNMT('opus-mt', max_loaded_models=10)
    file_data = []
    with open(src_path) as csvfile:
        print('src_path:' + src_path)
        csv_reader = csv.reader(csvfile)
        file_header = next(csv_reader)
        for row in csv_reader:
            file_data.append(row)
    for row in file_data:
        # print(row)
        text = row[1]
        #trans_list.append(text)
        result = trans_back(model, src, tar, text=text)
        #result = trans_back(model, text=text)
        if result == text:
            N_aug = N_aug + 1
        N = N + 1
        row[1] = result
    print("The number of N_aug: {}".format(N_aug))
    print("The number of N: {}".format(N))

    with open(tar_path, 'w', newline='') as csvfile:
        fieldnames = ['a_id', "a_text", "a_q_id", "score", "feedback"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for v in file_data:
            writer.writerow({'a_id':v[0], "a_text":v[1], "a_q_id": v[2], "score":v[3], "feedback":v[4]})


def load_data(data_dir, test_dir):
    train_files, test_files = [], []
    train_q_data, test_q_data = {}, {}
    train_a_data, test_a_data = {}, {}
    for path, dirnames, filenames in os.walk(data_dir):
      # print('{} {} {}'.format(repr(path), repr(dirnames), repr(filenames)))
      for file in filenames:
        if os.path.splitext(file)[1] == '.xml':
          file_path = path + "/" + file
          train_files.append(file_path)

    for path, dirnames, filenames in os.walk(test_dir):
      # print('{} {} {}'.format(repr(path), repr(dirnames), repr(filenames)))
      for file in filenames:
        if os.path.splitext(file)[1] == '.xml':
          file_path = path + "/" + file
          test_files.append(file_path)
    # Train
    n_q = 0
    n_a = 0
    for file in train_files:
      print(file)
      doc = xml4h.parse(file)
      # process question:
      q_id = doc.question.attributes['id']
      q_text = doc.question.questionText.text
      reference_answer_list = []
      rubric_list = []
      print("Processing Question # {} : {}".format(q_id, q_text))

      #This iterates through reference answers
      ref_ans_arr = []
      if isinstance(doc.question.referenceAnswers.referenceAnswer, list):
          ref_ans_arr = doc.question.referenceAnswers.referenceAnswer
      else:
          ref_ans_arr.append(doc.question.referenceAnswers.referenceAnswer)

      for ref_ans in ref_ans_arr:
          a_id = q_id + "_" + ref_ans.attributes['id']
          a_text = ref_ans.text
          a_q_id = q_id
          score = "correct"
          feedback = "correct"
          n_a += 1
          reference_answer_list.append(a_text)
          print(a_text)
          train_a_data[a_id]=[a_text, a_q_id, score, feedback]
      n_q += 1
      train_q_data[q_id]= [q_text, reference_answer_list, rubric_list]

      # This iterates through student answers

      stu_ans_arr = []
      if isinstance(doc.question.studentAnswers.studentAnswer, list):
          stu_ans_arr = doc.question.studentAnswers.studentAnswer
      else:
          stu_ans_arr.append(doc.question.studentAnswers.studentAnswer)

      for stu_ans in stu_ans_arr:
          a_id = stu_ans.attributes['id']
          a_text = stu_ans.text
          a_q_id = q_id
          score = stu_ans.attributes['accuracy']
          feedback = stu_ans.attributes['accuracy']
          n_a += 1
          train_a_data[a_id] = [a_text, a_q_id, score, feedback]

    # for id, v in train_q_data.items():
    #     print("qestion {}:  \n {}".format(id, v))
    # for id, v in train_a_data.items():
    #     print("answer {}:  \n {}".format(id, v))
    print("# of Train question in old is {}".format(n_q))
    print("# of Train question in new is {}".format(len(train_q_data)))
    print("# of Train answer in old is {}".format(n_a))
    print("# of Train answer in new is {}".format(len(train_a_data)))

    # Test
    n_q = 0
    n_a = 0
    for file in test_files:
        print(file)
        doc = xml4h.parse(file)
        # process question:
        q_id = doc.question.attributes['id']
        q_text = doc.question.questionText.text
        reference_answer_list = []
        rubric_list = []
        #print("Processing Question # {} : {}".format(q_id, q_text))

        # This iterates through reference answers
        ref_ans_arr = []
        if isinstance(doc.question.referenceAnswers.referenceAnswer, list):
            ref_ans_arr = doc.question.referenceAnswers.referenceAnswer
        else:
            ref_ans_arr.append(doc.question.referenceAnswers.referenceAnswer)

        for ref_ans in ref_ans_arr:
            a_id = q_id + "_" + ref_ans.attributes['id']
            a_text = ref_ans.text
            a_q_id = q_id
            score = "correct"
            feedback = "correct"
            n_a += 1
            reference_answer_list.append(a_text)
            test_a_data[a_id] = [a_text, a_q_id, score, feedback]

        n_q += 1
        test_q_data[q_id] = [q_text, reference_answer_list, rubric_list]

        # This iterates through student answers

        stu_ans_arr = []
        if isinstance(doc.question.studentAnswers.studentAnswer, list):
            stu_ans_arr = doc.question.studentAnswers.studentAnswer
        else:
            stu_ans_arr.append(doc.question.studentAnswers.studentAnswer)

        for stu_ans in stu_ans_arr:
            a_id = stu_ans.attributes['id']
            a_text = stu_ans.text
            a_q_id = q_id
            score = stu_ans.attributes['accuracy']
            feedback = stu_ans.attributes['accuracy']
            n_a += 1
            test_a_data[a_id] = [a_text, a_q_id, score, feedback]

    for id, v in train_q_data.items():
        print("qestion {}:  \n {}".format(id, v))
    # for id, v in test_a_data.items():
    #     print("answer {}:  \n {}".format(id, v))
    print("# of Test question in old is {}".format(n_q))
    print("# of Test question in new is {}".format(len(test_q_data)))
    print("# of Test answer in old is {}".format(n_a))
    print("# of Test answer in new is {}".format(len(test_a_data)))
    return train_q_data, train_a_data, test_q_data, test_a_data


def output_data(q_dict, a_dict, q_path, a_path):
    #wpath = os.getcwd() + q_path
    with open(q_path, 'w', newline='') as csvfile:
        fieldnames = ['q_id', "q_text", "reference_answer_list", "rubric_list"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for id, v in q_dict.items():
            writer.writerow({'q_id':id, "q_text":v[0], "reference_answer_list":v[1], "rubric_list":v[2]})

    with open(a_path, 'w', newline='') as csvfile:
        fieldnames = ['a_id', "a_text", "a_q_id", "score", "feedback"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for id, v in a_dict.items():
            writer.writerow({'a_id':id, "a_text":v[0], "a_q_id": v[1], "score":v[2], "feedback":v[3]})
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./example/train',
                        help='data_path')
    parser.add_argument('--test_path', type=str, default='./example/test',
                        help='data_path')
    parser.add_argument('--new_train_path', type=str,
                        default='./example/aug/train/',
                        help='data_directory')
    parser.add_argument('--new_test_path', type=str,
                        default='./example/aug/test/',
                        help='data_directory')
    parser.add_argument('--src_path', type=str, default='../example/new_format/train/aug_answer.csv',
                        help='data aug')
    parser.add_argument('--tar_path', type=str, default='../example/new_format/train/ch_aug_answer.csv',
                        help='data aug')
    parser.add_argument('--aug_src', type=str, default='en',
                        help='data aug')
    parser.add_argument('--aug_tar', type=str, default='zh',
                        help='data aug ')
    args = parser.parse_args()
    # train_q_data, train_a_data, test_q_data, test_a_data = load_data(args.train_path, args.test_path)
    # train_q_path = args.new_train_path + 'aug_question.csv'
    # train_a_path = args.new_train_path + 'aug_answer.csv'
    # test_q_path = args.new_test_path + 'aug_question.csv'
    # test_a_path = args.new_test_path + 'aug_answer.csv'
    # output_data(train_q_data, train_a_data, train_q_path, train_a_path)
    # output_data(test_q_data, test_a_data, test_q_path, test_a_path)
    aug_data(args.src_path, args.tar_path, args.aug_src, args.aug_tar)



if __name__ == "__main__":
    print("start ")
    main()
    print("done")