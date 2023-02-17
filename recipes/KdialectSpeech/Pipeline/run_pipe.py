#!/usr/bin/env python

'''
실행법 : python run_pipe.py run_pipe.yaml

speechbrain/utils/superpowers.py 사용 검토
'''

import logging
import sys
import os
import subprocess
import glob
import shutil

# from tqdm import tqdm
from pathlib import Path

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from kdialectspeech.resample import resample_audio

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    ##### setup logging
    logger = logging.getLogger(__name__)
    log_config = hparams["log_config"]
    log_file = hparams["log_file"]
    logger_overrides = {
        "handlers": {"file_handler": {"filename": log_file}}
    }
    # setup_logging(config_path="log-config.yaml", overrides={}, default_level=logging.INFO)
    sb.utils.logger.setup_logging(log_config, logger_overrides)
    #####

    #####
    ##### 방언별 실행 : 토크나이저, 언어모델, 음성인식모델
    #####

    run_provinces = hparams["run_provinces"]
    gpu_num = hparams["gpu_num"]
    pretrained_model_base = hparams['pretrained_model_base']

    for run_province in run_provinces:
        province_option = '--province_code=' + run_province
        pretrained_model_dir = os.path.join(pretrained_model_base, run_province)
        logger.info(f'make pretrained_model_dir : {pretrained_model_dir}')
        os.makedirs(pretrained_model_dir, exist_ok=True)

        ##### 토크나이저 실행(방언별로 각각 실행)
        # - 데이터 준비  
        # - 토큰화
        # subprocess 사용
        if 'tokenizer' in hparams['run_modules']:
            logger.info(f'tokenizer run_option : {province_option}')
            tokenizer_dir = hparams["tokenizer_dir"]
            logger.info(f'tokenizer_dir : {tokenizer_dir}')

            if run_province == 'jj':
                # cmd = ['python', 'train.py', 'hparams/1K_unigram_subword_bpe_jj.yaml --device=cpu' + province_option] # 리스트로는 실행이 안됨
                # cmd = 'python train.py hparams/1K_unigram_subword_bpe_jj.yaml --device=cpu ' + province_option
                cmd = 'python train.py hparams/1K_unigram_subword_bpe_jj.yaml ' + province_option
            else:
                # cmd = ['python', 'train.py', 'hparams/5K_unigram_subword_bpe.yaml', '--device=cpu', province_option] # 리스트로는 실행이 안됨
                # cmd = 'python train.py hparams/5K_unigram_subword_bpe.yaml --device=cpu ' + province_option
                cmd = 'python train.py hparams/5K_unigram_subword_bpe.yaml ' + province_option # tokenizer는 CPU에서 계산됨
            
            logger.info(f'cmd : {cmd}')
            result = subprocess.run(cmd,
                text=True,
                # capture_output=True,
                cwd=tokenizer_dir,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            logger.info(f'subprocess.CompletedProcess : {subprocess.CompletedProcess}')
            # if subprocess.CompletedProcess.returncode == 0:
            if result.returncode == 0:
                logger.info(f'tokenizer completed successfully')
                if hparams['copy_trained_model']:

                    token_result_dir = os.path.join(tokenizer_dir, 'results/data_prepared/' + run_province)
                    if run_province == 'jj': 
                        tokenizer =  os.path.join(token_result_dir, '1000_unigram.model')
                    else:
                        tokenizer =  os.path.join(token_result_dir, '5000_unigram.model')

                    # copy to pretrained_model_dir
                    tokenizer_target = os.path.join(pretrained_model_dir, 'tokenizer.ckpt')
                    logger.info(f'tokenizer file copy from {tokenizer} to {tokenizer_target}')
                    shutil.copy(tokenizer, tokenizer_target)

        else:
            logger.info(f'tokenizer is not in run_modules')


        ##### 언어모델
        # 언어모델 실행(방언별로 각각 실행)
        if 'lm' in hparams['run_modules']:
            logger.info(f'lm run_option : {province_option}')
            logger.info(f'gpu num : {gpu_num}')
            lm_dir = hparams["lm_dir"]
            if gpu_num > 1: # 제주도일 경우 처리(vocab size가 다름)
                cmd = "python -m torch.distributed.launch --nproc_per_node=" \
                    + str(gpu_num) \
                    + " train.py hparams/transformer.yaml --distributed_launch --distributed_backend='nccl' " \
                    + province_option
            else:
                cmd = 'python train.py hparams/transformer.yaml ' + province_option
            result = subprocess.run(
                cmd,
                text=True,
                # capture_output=True,
                cwd=lm_dir,
                shell=True,
                stdout=subprocess.PIPE
            )

            logger.info(f'subprocess.CompletedProcess : {subprocess.CompletedProcess}')
            # if subprocess.CompletedProcess.returncode == 0:
            if result.returncode == 0:
                logger.info(f'LM completed successfully')
                if hparams['copy_trained_model']:

                    # copy : asr model file for inference
                    lm_result_dir = os.path.join(lm_dir, 'results/Transformer/5555' + run_province + '/save')

                    lm_model_dir = sorted(glob.glob(lm_result_dir + '/CKPT*'), key=os.path.getmtime)[0]
                    lm_model_dir = Path(lm_model_dir).stem
                    lm_model = os.path.join(lm_result_dir, lm_model_dir + 'model.ckpt')
                    lm_target = os.path.join(pretrained_model_dir, 'lm.ckpt')
                    logger.info(f'lm file copy from {lm_model} to {lm_target}')
                    shutil.copy(lm_model, lm_target)

        else:
            logger.info(f'lm is not in run_modules')

        #####

        # 음성인식 실행(방언별로 각각 실행)  
        # 추론 준비 : 추론에 필요한 모델 파일을 한 디렉토리에 복사

        ##### ASR
        if 'asr' in hparams['run_modules']:
            logger.info(f'asr run_option : {province_option}')
            logger.info(f'gpu num : {gpu_num}')
            logger.info(f'province : {run_province}')
            asr_dir = hparams["asr_dir"]
            print(asr_dir)
            if gpu_num > 1: # 제주도일 경우 처리(vocab size가 다름)
                # GPU 지정하여 사용할 경우
                # export CUDA_VISIBLE_DEVICES="1,2,3"
                # 위 환경을 python에서 지정
                # os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
                cmd = "python -m torch.distributed.launch --nproc_per_node=" \
                    + str(gpu_num) \
                    + " train.py hparams/conformer_medium.yaml --distributed_launch --distributed_backend='nccl' " \
                    + province_option
            else:
                cmd = 'python train.py hparams/conformer_medium.yaml ' + province_option
            result = subprocess.run(
                # ['python train.py hparams/conformer_medium.yaml', '--province_code=gs'],
                # ['python train.py hparams/conformer_medium.yaml', province_option],
                [cmd, province_option],
                # python -m torch.distributed.launch --nproc_per_node=4 train.py hparams/conformer_medium.yaml --distributed_launch --distributed_backend='nccl'
                text=True,
                # capture_output=True,
                cwd=asr_dir,
                shell=True,
                stdout=subprocess.PIPE
            )

            logger.info(f'subprocess.CompletedProcess : {subprocess.CompletedProcess}')
            # if subprocess.CompletedProcess.returncode == 0:
            if result.returncode == 0:
                logger.info(f'asr completed successfully')
                if hparams['copy_trained_model']:

                    # copy :  hparam file for inference
                    hparam_file = os.path.join(pretrained_model_base, 'hyperparams.yaml')
                    hparam_file_target = os.path.join(pretrained_model_dir, 'hyperparams.yaml')
                    logger.info(f'hparma file copy from {hparam_file} to {hparam_file_target}')
                    shutil.copy(hparam_file, hparam_file_target)

                    # copy : asr model file for inference
                    train_result_dir = os.path.join(asr_dir, 'results/Conformer/5555/' + run_province + '/save')

                    best_model_dir = sorted(glob.glob(train_result_dir + '/CKPT*'), key=os.path.getmtime)[0]
                    best_model_dir = Path(best_model_dir).stem
                    best_model = os.path.join(train_result_dir, best_model_dir, 'model.ckpt')
                    best_model_target = os.path.join(pretrained_model_dir, 'asr.ckpt')
                    logger.info(f'asr model file copy from {best_model} to {best_model_target}')
                    shutil.copy(best_model, best_model_target)

                    # copy : normalizer file for inference
                    normalizer = os.path.join(train_result_dir, best_model_dir + '/normalizer.ckpt')
                    normalizer_target = os.path.join(pretrained_model_dir, 'normalizer.ckpt')
                    logger.info(f'normalizer file copy from {best_model} to {best_model_target}')
                    shutil.copy(normalizer, normalizer_target)

                    # os.symlink(best_model, lm_model)

        else:
            logger.info(f'asr is not in run_modules')

        #####
