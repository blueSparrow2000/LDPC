from prettyprinter import *
from ECO import *
from LDPC_sampler import *
from extracter import *
from verifier import *
from variables import *
from GERCE import GERCE
from formatter import *

import time
start_time = time.time()



H, A = sample_LDPC(codeword_len,databit_num,pooling_factor=pooling_factor,noise_level=noise_level)

if not LARGE_CODE:
    print("H matrix: ")
    print_arr(H)

    # print("Code word matrix: ")
    # print_arr(A)
else:
    print("Code word generated")

print("Elapsed time: %s seconds" % round(time.time() - start_time,3))


# Testing GERCE
# with FULL column extractions

Hr = GERCE(A)

print("GERCE complete!")
this_time = time.time()
print("Elapsed time: %s seconds" % round(this_time - start_time,3))

print("Success?: ", check_success(H,Hr))



'''
format해야 정확도가 올라간다! 형식에 맞추는게 중요함...
간단한 예시의 경우, 오른쪽 (n-k)x(n-k) 부분이 identity라는 점을 이용함! 
format하기 전에는 4/100 개 찾음 => format후 71/100개를 찾음. 모두 총 92개의 dual vector를 찾은 경우.


이것이 의미하는 바는, QC LDPC 형식이라는 것을 알때 구한 PCM을 QC LDPC형식에 맞도록 하면 
정확도를 크게 향상시킬 수 있음을 의미한다! 
즉, QC ldpc 변환 함수를 만들어 두는게 필요함!

'''
