# 🎥 AI 기반 영상 속 특정 인물 제거 프로젝트  
*(FaceID + Tracking + Inpainting Pipeline)*

모델은 별도 저장소에서 관리하며,  
본 프로젝트는 **영상 속 특정 인물을 자동으로 식별·추적·제거하는 시스템**을 구현하는 것을 목표로 합니다.

---

## 📌 프로젝트 개요

본 프로젝트는 AI 기술을 활용하여 영상에 등장하는 인물 중  
**사용자가 선택한 특정 인물만을 자동으로 제거**하는 영상 처리 시스템입니다.

영상에 등장하는 모든 얼굴을 인식한 뒤 Face ID 단위로 분류하고,  
선택된 인물에 대해 객체 세그멘테이션과 트래킹을 수행하여  
Inpainting 모델을 통해 자연스럽게 배경을 복원합니다.

비전문가도 몇 번의 클릭만으로 영상 속 특정 인물을 제거할 수 있도록  
전체 과정을 자동화한 것이 특징입니다.

---

## 🧠 문제 의식

- 영상에서 특정 인물만 제거하려면 프레임 단위 수작업이 필요하여 시간이 오래 걸림  
- 단순 객체 제거는 가능하나, **특정 인물만 지속적으로 추적하여 제거하는 기능은 구현 난이도가 높음**

---

## 🛠 핵심 아이디어

- 얼굴 벡터 유사도를 기반으로 **Face ID를 생성하여 인물을 분류**  
- 사용자가 제거할 인물을 선택하면 해당 인물만 추적  
- 세그멘테이션 + 트래킹 + 인페인팅을 하나의 파이프라인으로 구성하여 자동화  

---

## ⚙️ 처리 파이프라인

1. InsightFace로 영상 내 모든 얼굴 검출 및 임베딩 벡터 추출  
2. 벡터 유사도 기반 Face ID 생성 및 인물 분류  
3. 사용자 선택 Face ID 기반 대상 인물 지정  
4. Detectron2로 사람 객체 세그멘테이션 수행  
5. Tracking 모델로 대상 인물 추적  
6. 추적된 마스크를 Inpainting 모델에 전달  
7. 선택 인물 제거 및 배경 복원 영상 생성  

---

## 🖥 실행 화면

### 1️⃣ 영상 업로드
<p align="center">
  <img src="https://github.com/user-attachments/assets/286eb9d5-c994-4fef-bf42-98f7c5a2733a" width="800" alt="영상 업로드 화면"/>
</p>

### 2️⃣ 얼굴 선택 (Face ID)
<p align="center">
  <img src="https://github.com/user-attachments/assets/1f4e77c7-6939-4061-9b2f-4699bd293dee" width="800" alt="얼굴 선택 화면"/>
</p>

### 3️⃣ 객체 선택 (세그멘테이션 대상)
<p align="center">
  <img src="https://github.com/user-attachments/assets/959ea4d8-fbe6-4b13-a13c-39ae6d766ce9" width="800" alt="객체 선택 화면"/>
</p>

### 4️⃣ 세그멘테이션 결과 비교 (원본 vs 처리)
<p align="center">
  <img src="https://github.com/user-attachments/assets/f38a8c65-5b3e-4638-a934-74a937eee1fd" width="800" alt="세그멘테이션 결과 비교"/>
</p>

### 5️⃣ 인페인팅 결과 비교 (원본 vs 처리)
<p align="center">
  <img src="https://github.com/user-attachments/assets/9b60fb1d-366b-46cb-abe9-77abc11e2499" width="800" alt="인페인팅 결과 비교"/>
</p>

---

### 🔹 정방향 트래킹만 적용한 결과 (정확도 낮음)
대상 인물이 화면 밖으로 이탈했다가 다시 등장하는 구간에서  
트래킹이 끊기며 마스크가 누락되는 현상이 발생합니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/da17b8b9-db86-4722-b810-7bb631f39ff1" width="820" alt="정방향 트래킹 결과 (정확도 낮음)"/>
</p>

---

### 🔹 양방향 트래킹 적용 후 결과 (정확도 개선)
정방향 트래킹과 역방향 트래킹을 결합하여  
재등장 구간에서도 추적이 유지되며 제거 결과가 안정적으로 출력됩니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/535e4581-24e3-476b-adac-baec6f9df0ac" width="820" alt="양방향 트래킹 결과 (정확도 개선)"/>
</p>

---

## 🔧 트러블슈팅

### 문제
정방향(Forward) 트래킹만 적용했을 때,  
대상 인물이 **화면 밖으로 잠시 이탈했다가 다시 화면에 재등장하는 구간**에서  
트래킹이 끊기며 마스크가 누락되는 현상이 발생했습니다.

### 해결
- 영상 프레임을 역순으로 처리하는 **역방향(Backward) 트래킹**을 추가 적용  
- 정방향 트래킹 결과와 역방향 트래킹 결과를 결합하여  
  재등장 구간에서도 추적이 이어지도록 보완  
- 전체 구간에서의 추적 안정성과 정확도를 개선  

---

## 🛠 사용 기술 및 라이브러리

- **Python**  
- **Flask** (Web Server, REST API)  
- **InsightFace** (Face Embedding, Face ID 분류)  
- **Detectron2** (Person Segmentation)  
- **Tracking Model** (Forward / Backward Tracking)  
- **Inpainting Model** (배경 복원)  
- **OpenCV** (프레임 처리)  
- **MoviePy** (영상 입출력)  
