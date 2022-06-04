
딥러닝(pytorch)과 플라스크를 활용한 연예인 이미지 유사도 측정 사이트



필요 사항


1. cv2(opencv-python) modeul 필요

2. pytorch




사용법


1. app.py를 플라스크로 구동


2. templates 폴더 내의 'faceform.html' 페이지를 통해 사용자가 이미지를 업로드


3. 이미지는 app.py에서 request되어 app.py에서 'get_prediction'함수 적용

    3-1. 함수의 자세한 사항은 'ai_logic.ipynb' 참조

    3-2. request된 이미지는 'upload'폴더에 임시저장

    3-3. 임시저장된 이미지는 이미지 내의 사람 얼굴만 인식, crop해 'crop'폴더에 임시저장

    3-4. crop된 임시 이미지를 'model.pt'파일의 model을 적용, 닮은 연예인의 index를 retrun

    3-5. 'name.txt'파일 내의 연예인 리스트를 토대로 return된 index 값을 적용


4. 'faceresult.html' 페이지에서 결과값과 함께

   'static/images/' 폴더 내에서 결과값과 동일한 이름을 가진 이미지(.jpg)를 load




유의사항:


1. 연예인 목록을 추가, 삭제하기 위해서는 새로 학습 시켜야함.


1-1. 따라서 'static/images/'폴더 내 이미지 파일과 index는 반드시 modle과 일치해야 함.
    
    
2. 'static/images/' 폴더 내 이미지는 반드시 '.jpg' 형식 파일만 저장


2. 구글에서 이름을 검색했을 때, 다른 이미지가 나오는 연예인은 가능한한 추가하지 않는 것을 권장


3. 동명이인은 추가하면 안 됨


4. 선글라스, 마스크로 인해 이목구비 인식이 힘든 이미지는 사용 불가능

5. 'model.pt' 파일을 강제로 실행하면 에러 발생




version:

2022-06-03 / 12:00

구버전 model.pt 사용 중
연예인 list = 232명

2022-06-03 / 14:30

신버전 model.pt 사용
연예인 list = 231명 (김태원 삭제)


2022-06-03 / 17:30

image 폴더 내의 사진 목록은 추가되었으나, 아직 모델 학습 미실시로 모델 구동 안 됨
연예인 list = 538명


* 반드시 김태원 삭제한 채로 모델 학습 실시할 것
