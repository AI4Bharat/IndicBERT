from tqdm import tqdm

def read_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return lines

sents_file = "/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_sents/as.txt"
target_file = "/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/as.txt"

src_file = "/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/lang-wise-src/as.txt"
tgt_file = "/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/lang-wise-tgt/as.txt"

sents = read_file(sents_file)
all_src = read_file(src_file)
all_tgt = read_file(tgt_file)

dict_ = {k:v for k,v in zip(all_src, all_tgt)}

target_lines = []
for s in sents:
    if s == '\n':
        target_lines.append('\n')
    elif s in all_src.keys():
        target_lines.append(dict_[s])

with open(target_file, 'w') as f:
    for t in target_lines:
        f.write(t)