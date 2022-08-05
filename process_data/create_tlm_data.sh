# translation data creation
create_tlm_data () {
   python create_mlm_data.py \
        --input_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/en-$1 \
        --output_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/translated_paras/en-$1.tfrecord \
        --input_file_type=parallel \
        --vocab_file=/nlsasfs/home/ai4bharat/gramesh/bertteam/IndicXLM/data/gcp/wordpiece_250k/vocab.txt \
        --do_lower_case=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=77 \
        --do_whole_word_mask=True \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=$2
} 

create_tlm_data as 9
create_tlm_data bn 2
create_tlm_data gu 2
create_tlm_data hi 1
create_tlm_data kn 2
create_tlm_data ml 2
create_tlm_data mr 2
create_tlm_data or 6
create_tlm_data pa 3
create_tlm_data ta 2
create_tlm_data te 2