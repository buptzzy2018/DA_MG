#!/bin/bash
# Copyright   2021   Tsinghua University (Author: Lantian Li, Yang Zhang)
# Apache 2.0.


SUNINE_ROOT=../..
cnceleb1_path=/ssd.m2/CNC_v2.0/CN-Celeb
cnceleb2_path=/ssd.m2/CNC_v2.0/CN-Celeb2
musan_path=/ssd.m2/musan
rirs_path=/ssd.m2/RIRS_NOISES

config=conf/baseline.yaml
exp_dir="exp/resnet_asp_aam"
ckpt_path=
cuda_device=2,3  #change --gpus correspondingly if you change the device
stage=4
stop_stage=4


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ];then
  # flac to wav
  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/data \
          --speaker_level 1

  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb2_path/data \
          --speaker_level 1
 
  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/eval/enroll \
          --speaker_level 0
  
  python3 $SUNINE_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/eval/test \
          --speaker_level 0
fi


# In our experiment, we found that VAD seems useless.
# Here directly skip this stage.
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ];then
  # compute VAD for each dataset
  echo Compute VAD on cnceleb1
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/data \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 16

  echo Compute VAD on cnceleb2
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb2_path/data \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 16

  echo Compute VAD on enroll
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/eval/enroll \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 16

  echo Compute VAD on test
  python3 $SUNINE_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/eval/test \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 16
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ];then
  # prepare data

  combine_short_audio=1

  if [ -d data ]; then
    rm -r data
  fi
  mkdir -p data/wav/train

  if [ $combine_short_audio -eq 1 ];then
    echo "combine short audios ..."
    bash local/comb_cn1_cn2.sh \
        --cnceleb1_audio_dir ${cnceleb1_path}/data/ \
        --cnceleb2_audio_dir ${cnceleb2_path}/data/ \
        --min_duration 5 \
        --get_dur_nj 60 \
        --statistics_dir data/statistics \
        --store_data_dir data/wav/train
  else
    for spk in `cat ${cnceleb1_path}/dev/dev.lst`; do
      ln -s ${cnceleb1_path}/data/${spk} data/wav/train/$spk
    done

    for spk in `cat ${cnceleb2_path}/spk.lst`; do
      ln -s ${cnceleb2_path}/data/${spk} data/wav/train/$spk
    done
  fi

  echo "prepare training data ..."
  mkdir -p data/train

  for spk in data/wav/train/*; do
    find ${spk}/ -name "*.wav" | \
    awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort >> data/train/wav.scp
  done

  awk '{print $1}' data/train/wav.scp | awk -F "/" '{print $0,$1}' > data/train/utt2spk
  local/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt

  echo "prepare testing data ..."
  mkdir -p data/eval

  find ${cnceleb1_path}/eval/ -name "*.wav" | awk -F"/" '{print $(NF-1)"/"$NF,$0}' | sort > data/eval/wav.scp
  awk '{print $1}' data/eval/wav.scp | awk -F "[/-]" '{print $0,$2}' > data/eval/utt2spk

  awk '{print $0}' ${cnceleb1_path}/eval/lists/enroll.map | \
  awk -v p=${cnceleb1_path}/data '{for(i=2;i<=NF;i++){print $i, p"/"$i}}' > data/eval/enroll.scp
  cat data/eval/enroll.scp >> data/eval/wav.scp
  awk '{print $1}' data/eval/enroll.scp | awk -F "/" '{print $0,$1"-enroll"}' >> data/eval/utt2spk
  cp ${cnceleb1_path}/eval/lists/enroll.map data/eval/enroll.map

  echo "prepare evaluation trials ..."
  mkdir -p data/trials
  python3 local/format_trials_cnceleb.py \
          --cnceleb_root $cnceleb1_path \
          --dst_trl_path data/trials/CNC-Eval-Core.lst
  
  awk '{print $3,$1,$2}' ${cnceleb1_path}/eval/lists/trials.lst > data/trials/CNC-Eval-Avg.lst
fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ];then
  # prepare data for model training

  mkdir -p data

  echo Build train list

  python3 $SUNINE_ROOT/steps/build_datalist.py \
          --data_dir data/wav/train \
          --extension wav \
          --speaker_level 1 \
          --data_list_path data/train_lst.csv

  echo Build $musan_path list
  python3 $SUNINE_ROOT/steps/build_datalist_aug.py \
          --data_dir $musan_path \
          --extension wav \
          --data_list_path data/musan_lst.csv

  echo Build $rirs_path list
  python3 $SUNINE_ROOT/steps/build_datalist_aug.py \
          --data_dir $rirs_path \
          --extension wav \
          --data_list_path data/rirs_lst.csv
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ];then
  # model training
  echo "model training ..."
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SUNINE_ROOT/main.py \
          --config $config \
          ${ckpt_path:+--checkpoint_path $ckpt_path} \
          --exp_dir $exp_dir \
          --train_list_path data/train_lst.csv \
          --musan_list_path data/musan_lst.csv \
          --rirs_list_path data/rirs_lst.csv \
          --eval_list_path data/eval/wav.scp \
          --trials_path data/trials/CNC-Eval-Core.lst \
          --distributed_backend dp \
          --reload_dataloaders_every_epoch \
          --gpus 2

fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ];then
  # average checkpoints

  echo "average checkpoints ..."
  avg_model=$exp_dir/checkpoints/avg_model.ckpt
  last_n=10  # Change the `last_n` parameter of config.yaml correspondingly

  python $SUNINE_ROOT/steps/average_checkpoints.py \
      --src_path $exp_dir/checkpoints \
      --dest_model $avg_model \
      --last_n $last_n
fi


if [ $stage -le 6 ] && [ $stop_stage -ge 6 ];then
  echo "extract embedding ..."

  ckpt_path=$exp_dir/checkpoints/avg_model.ckpt
  echo $ckpt_path

  for dset in train eval; do
    xvec_path=$exp_dir/embeddings/$dset
    [ -d $xvec_path ] && rm -r $xvec_path
    mkdir -p $xvec_path
    
    local/extract.sh \
        --SUNINE_ROOT $SUNINE_ROOT \
        --config $config \
        --exp_dir $exp_dir \
        --ckpt_path $ckpt_path \
        --xvec_path $xvec_path \
        --eval_scp_path data/$dset/wav.scp \
        --cuda_device $cuda_device
  done

  # mean vector of enroll
  echo "mean vector of enroll"
  python $SUNINE_ROOT/steps/vector_mean.py \
    --spk2utt data/eval/enroll.map \
    --xvector_scp $exp_dir/embeddings/eval/xvector.scp \
    --spk_xvector_ark $exp_dir/embeddings/eval/enroll_spk_xvector.ark

  cat $exp_dir/embeddings/eval/enroll_spk_xvector.scp >> $exp_dir/embeddings/eval/xvector.scp
fi


if [ $stage -le 7 ] && [ $stop_stage -ge 7 ];then
  # evaluation

  mkdir -p $exp_dir/scores/

  for trials in CNC-Eval-Core CNC-Eval-Avg; do
    echo Evaluate $trials
    python -W ignore $SUNINE_ROOT/trainer/metric/compute_score.py \
            --trials_path data/trials/$trials.lst \
            --eval_scp_path $exp_dir/embeddings/eval/xvector.scp \
            --scores_path $exp_dir/scores/$trials.foo
  done
fi



