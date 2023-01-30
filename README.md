# kdialectspeech
AI Hub 중노년층 방언 데이터셋입니다. 이 페이지는 중노년층 방언 데이터를 이용한 음성인식 모델을 학습하는 프로그램 저장소입니다.
kdialectspeech 모델 학습은 [Speechbrain](https://speechbrain.github.io) Toolkit을 이용합니다. Speechbrain은 파이토치를 기반으로 하는 음성언어용 올인원 AI 도구이며 오픈소스입니다. 스피치브레인에 대한 자세한 내용은 아래의 공식 홈페이지를 참고하시기 바랍니다.  
<p align="center">
    <a href='https://speechbrain.github.io'>
        <img src="docs/images/speechbrain-logo.svg" alt="SpeechBrain Logo"/>
    </a>
</p>

## Install
kdialectspeech 데이터를 이용한 음성인식 모델링을 실행하기 위해서는 KdialectSpeech 저장소에서 소스를 다운로드 받아서 필요한 환경을 설치합니다.  
먼저 Python 3.7+을 사용할 수 있는 환경을 준비 합니다.  
그리고 다음의 명령을 실행하여 스피치브레인을 설치합니다.
```
git clone https://github.com/starcell/KdialectSpeech.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```
스피치브레인 저장소의 소스는 kdialectspeech 모델링과 맞지 않는 부분이 있을 수 있으므로 KdialectSpeech 저장소의 소스를 다운로드 받아서 사용할 것을 권장합니다.


파이썬에서 아래와 같이 스피치브레인 임포트가 되는 지 확인합니다.
```
import speechbrain as sb
```
오류가 없이 임포트가 되면 스피치브레인이 정상적으로 설치된 것입니다.  

데이터 준비를 위하여 ffmpeg가 필요하므로 아래와 같이 명령을 실행하여 설치합니다.
```
apt update
apt install ffmpeg
```

## Test Installation
다음과 같은 명령을 실행하여 설치된 내용들이 정상적으로 작동하는지 확인합니다.
```
pytest tests
pytest --doctest-modules speechbrain
```

# Running an experiment
이제 KdialectSpeech 모델링 학습을 실행합니다.

```
> cd recipes/KialectSpeech/Pipeline/
> python run_pipe.py run_pipe.yaml
```

# License
KdialectSpeech 모델링 소스는 아파치 라이센스 버전 2.0을 따릅니다. KdialectSpeech가 사용하는 SpeechBrain AI Toolkit도 Apache License, version 2.0 하에 출시되었습니다. 아파치 라이센스는 대중적인 BSD계열의 라이센스입니다. 이 라이센스하에 출시된 SpeechBrrain과 KdialectSpeech는 라이센스 해더를 포함하면 자유롭게 재 비포할 수 있습니다. 자세한 내용은 LICENSE 파일을 참고하십시오.

# Citing SpeechBrain
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```

