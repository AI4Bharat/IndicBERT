# convert paras to sents
for lang in as bd bn dg en gom gu hi kha kn ks mai ml mni mr ne or pa sa sat sd ta te ur
do
python scripts/convert_para2sent.py --input ../monolingual_data/$lang.txt --output ../monolingual_sents/$lang.txt --lang $lang
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
python /home/cs21d409_cse_iitm_ac_in/IndicBERT/process_data/create_mlm_data.py \
    --input_file=as.txt, bd.txt, bn.txt, dg.txt, en.txt, gom.txt, gu.txt, hi.txt, kha.txt, kn.txt, ks.txt, mai.txt, ml.txt, mni.txt, mr.txt, ne.txt, or.txt, pa.txt, sa.txt, sat.txt, sd.txt, ta.txt, te.txt, ur.txt  \
    --output_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/tfrecords/eval.tfrecord \
    --vocab_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/wordpiece_250k/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=77 \
    --do_whole_word_mask=True \
    --masked_lm_prob=0.15 \
    --random_seed=12345

python /home/cs21d409_cse_iitm_ac_in/IndicBERT/process_data/create_mlm_data.py \
    --input_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/sample_text.txt \
    --mlm_output_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/sample_mlm.tfrecord \
    --tlm_output_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/sample_tlm.tfrecord \
    --parallel_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/sample_data/en-as,/home/cs21d409_cse_iitm_ac_in/IndicBERT/sample_data/en-bn,/home/cs21d409_cse_iitm_ac_in/IndicBERT/sample_data/en-hi \
    --vocab_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/wordpiece_250k/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=77 \
    --do_whole_word_mask=True \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor 5 \
    --num_workers 1

# create tfrecord for training data
python /home/cs21d409_cse_iitm_ac_in/IndicBERT/process_data/create_mlm_data.py \
    --input_file=  \
    --output_file= \
    --vocab_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/wordpiece_250k/vocab.txt \
    --do_lower_case=True \
    --max_seq_length=512 \
    --max_predictions_per_seq=77 \
    --do_whole_word_mask=True \
    --masked_lm_prob=0.15 \
    --random_seed=12345


# train MLM and TLM
python /home/cs21d409_cse_iitm_ac_in/IndicBERT/train/run_pretraining.py \
    --input_file=gs://indic-bert/test_tpu/input/test_examples.tfrecord \
    --output_dir=gs://indic-bert/large_tpu_test/ \
    --do_train=True \
    --bert_config_file=/home/cs21d409_cse_iitm_ac_in/IndicBERT/config.json \
    --train_batch_size=4096 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --num_train_steps=100000 \
    --num_warmup_steps=10000 \
    --learning_rate=2e-5 \
    --save_checkpoint_steps=10000 \
    --use_tpu=True \
    --tpu_name=indic-bert \
    --tpu_zone=us-east1-d \
    --num_tpu_cores=128