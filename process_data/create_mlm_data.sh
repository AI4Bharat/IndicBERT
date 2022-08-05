# monolingual data creation
create_mlm_data () {
   python create_mlm_data.py \
        --input_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_sents/$1.txt \
        --output_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/monolingual_sents/$1.tfrecord \
        --input_file_type=monolingual \
        --vocab_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/wordpiece_250k/vocab.txt \
        --do_lower_case=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=77 \
        --do_whole_word_mask=True \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=$2
} 

create_mlm_data as 9
create_mlm_data bd 40
create_mlm_data bn 2
create_mlm_data dg 148
create_mlm_data en 1
create_mlm_data gom 10
create_mlm_data gu 2
create_mlm_data hi 1
create_mlm_data kha 10
create_mlm_data kn 2
create_mlm_data ks 249
create_mlm_data mai 6
create_mlm_data ml 2
create_mlm_data mni 88
create_mlm_data mr 2
create_mlm_data ne 2
create_mlm_data or 6
create_mlm_data pa 3
create_mlm_data sa 5
create_mlm_data sat 43
create_mlm_data sd 44
create_mlm_data ta 2
create_mlm_data te 2
create_mlm_data ur 3