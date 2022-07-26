# convert paras to sents
for lang in as bd bn dg en gom gu hi kha kn ks mai ml mni mr ne or pa sa sat sd ta te ur
do
python scripts/convert_para2sent.py --input ../monolingual_data/$lang.txt --output ../monolingual_sents/$lang.txt
echo $lang
done


# create tokenizer
python tokenization/build_tokenizer.py --input ../tokenizer_data/ --output ../wordpiece_250k/ --vocab 250000


# sbatch command to create mlm data
sbatch --job-name mlm_data --gres gpu:0 -p cpup --cpus-per-task 128 --nodes 1 \
    --ntasks-per-node 1 --time=07-00:00:00 \
    --wrap 'srun --output mlm_data.log.node%t --error mlm_data.stderr.node%t.%j  \
    bash create_data.sh'
    
# create tfrecords for evalutation data
python ../IndicBERT/process_data/create_mlm_data.py \
    --input_file=as.txt, bd.txt, bn.txt, dg.txt, en.txt, gom.txt, gu.txt, hi.txt, kha.txt, kn.txt, ks.txt, mai.txt, ml.txt, mni.txt, mr.txt, ne.txt, or.txt, pa.txt, sa.txt, sat.txt, sd.txt, ta.txt, te.txt, ur.txt  \
    --output_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/tfrecords/eval.tfrecord \
    --vocab_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/wordpiece_250k/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=77 \
    --do_whole_word_mask=True \
    --masked_lm_prob=0.15 \
    --random_seed=12345