import random
import time
import sys
import itertools

# lang = sys.argv[1] 
		
# map_ = {
#     'as': 5,
#     'bd': 13,
#     'bn': 2,
#     'dg': 33,
#     # 'en': 1,
#     'gom': 5,
#     'gu': 2,
#     # 'hi': 1,
#     'kha': 5,
#     'kn': 2,
#     'ks': 49,
#     'mai': 4,
#     'ml': 2,
#     'mni': 29,
#     'mr': 2,
#     'ne': 2,
#     'or': 4,
#     'pa': 2,
#     'sa': 7,
#     'sat': 15,
#     'sd': 15,
#     'ta': 2,
#     'te': 2,
#     'ur': 2
# }

map_ = {
    'as': 8,
    'bd': 37,
    'bn': 2,
    'dg': 135,
    # 'en': 1,
    'gom': 10,
    'gu': 2,
    # 'hi': 1,
    'kha': 9,
    'kn': 2,
    'ks': 242,
    'mai': 6,
    'ml': 2,
    'mni': 110,
    'mr': 2,
    'ne': 2,
    'or': 6,
    'pa': 3,
    'sa': 16,
    'sat': 46,
    'sd': 43,
    'ta': 2,
    'te': 2,
    'ur': 3
}

for lang in list(map_.keys()):
    documents = []
    with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_paras/tagged-{lang}.txt', 'r') as f:
        lines = f.readlines()

    single_doc = []
    for line in lines:
        if line == '\n':
            documents.append(single_doc)
            single_doc = []
        else:
            single_doc.append(line)

    new = documents * map_[lang]

    start = time.time()
    random.shuffle(new)
    end = time.time()

    print(f'Time to shuffle: {end-start}')

    with open(f'/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_paras/temp-sampled-0.7-{lang}.txt', 'w') as f:
        for doc in new:
            for i, sent in enumerate(doc):
                f.write(sent.strip()+'\n')
            f.write('\n')