import sys
from absl import logging
import pandas
import codecs

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        print('Usage: python {} train_csv vocab_file\n'
              '    train_csv : csv file contain train dataset\n'
              '    vocab_file  : vocab file'.format(sys.argv[0]))
        exit(1)

    train_csv = sys.argv[1]
    vocab_file = sys.argv[2]
    char_dict = {'<unk>': 0, '<space>': 0}
    texts = pandas.read_csv(train_csv, sep='\t', usecols=['transcript'])
    for text in texts.itertuples():
        t = getattr(text, 'transcript')
        items = t[0].strip().split(" ")
        for item in items:
            for char in item:
                if char in char_dict:
                    char_dict[char] += 1
                else:
                    char_dict[char] = 1
    
    idx = 0
    with codecs.open(vocab_file, "w", "utf-8") as f:
        for key in char_dict.keys():
            char_id_str = key + ' ' + str(idx) + '\n'
            f.write(char_id_str)
            idx += 1
    print("Finished created vocab")






