import preprocess_module as lib
from glob import glob
import os
from tqdm import tqdm

# root라는 변수의 이름을 갖는 하위 폴터에서 모든 파일을 읽어들여서 image 파일로 변환함
# is_impact 꼭 확인!
# is_impact = True 이면 image/impact에, 그렇지 않으면 image/not_impact에 저장됨
root = "HB5_LPMS_False Alarm/**"
is_impact = False
lib.binary_file_eraser(root)
for file_name in tqdm(glob(root, recursive=True)):
    if os.path.isfile(file_name):
        lib.csv2img(file_name=file_name, is_impact=is_impact)