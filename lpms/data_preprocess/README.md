```HB5_LPMS_False Alarm```와 ```HB5_LPMS_Impact Test```는 데이터가 담겨있는 폴더입니다. csv형태로 충격과 비충격 데이터가 저장되어 있습니다.   
 
HB5_LPMS_False Alarm/   
├─ sub_direct1/   
│  ├─ file1.csv   
├─ sub_direct2/   
│  ├─ file2.csv   
HB5_LPMS_Impact Test/   
├─ sub_direct3/   
│  ├─ file3.csv/   
├─ sub_direct4/   
│  ├─ file4.csv   


```make_data.py```: 데이터의 폴더 구조를 유지하면서 csv파일을 이미지(png) 형태로 변환해주는 프로그램입니다.

image/   
├─ impact/   
│  ├─ HB5_LPMS_Impact Test/   
│  │  ├─ sub_direct3/   
│  │  │  ├─ file3/   
│  │  │  │  ├─ V-102.png   
│  │  │  │  ├─ V-101.png/   
│  │  ├─ sub_direct4/   
│  │  │  ├─ file4/   
│  │  │  │  ├─ V-101.png   
│  │  │  │  ├─ V-102.png   
├─ not_impact/   
│  ├─ HB5_LPMS_False Alarm/   
│  │  ├─ sub_direct1/   
│  │  │  ├─ file1/   
│  │  │  │  ├─ V-102.png   
│  │  │  │  ├─ v-101.png   
│  │  ├─ sub_direct2/   
│  │  │  ├─ file2/   
│  │  │  │  ├─ V-102.png   
│  │  │  │  ├─ v-101.png   

```flattening_directory.py```: 동작은 아래 예시를 참고하세요. image 폴더를 data라는 폴더로 복사하여 directory 구조와 파일명을 변형합니다.

data/   
├─ impact/   
│  ├─ HB5_LPMS_Impact_Test_sub_direct3_file3
│  │   ├─ V-101.png   
│  │   ├─ V-102.png
├─ not_impact/   
│  ├─ HB5_LPMS_False_Alarm_sub_direct1_file1
│  │   ├─ V-102.png
│  │   ├─ V-101.png

그외의 jupyter노트북은 위 프로그램을 짤때 사용했던 연습장입니다.