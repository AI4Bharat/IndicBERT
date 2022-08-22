# combine all langauges into 1 file + shuffle + split by N(5M) documents into multiple sub docs each with all language content

import random
import time

langs_str = "as bd bn dg en gom gu hi kha kn ks mai ml mni mr ne or pa sa sat sd ta te ur"
langs = langs_str.split()
print(f'Langs: {langs}')

documents = []
for lang in langs:
    with open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/temp-sampled-0.7/temp-sampled-0.7-{lang}.txt', 'r') as f:
        lines = f.readlines()

    print(f'Finished reading {lang}')
    single_doc = []
    for line in lines:
        if line == '\n':
            documents.append(single_doc)
            single_doc = []
        else:
            single_doc.append(line)

print(f'Length of docs: {len(documents)}')

start = time.time()
random.shuffle(documents)
end = time.time()

print(f'Time to shuffle: {end-start}')

doc_count = 0
with open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/splits-0.7/shuf-{doc_count//5000000}.txt', 'w') as f:
    for doc in documents:
        doc_count += 1
        for sent in doc:
            f.write(sent.strip()+'\n')
        f.write('\n')

        if doc_count % 5000000 == 0:
            f = open(f'/nlsasfs/home/ai4bharat/gramesh/fine-tuning/splits-0.7/shuf-{doc_count//5000000}.txt', 'w')