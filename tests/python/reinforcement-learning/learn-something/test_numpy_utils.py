
import numpy as np
from read_ncu_profile_excel import average_record

def test_average_record():
    a = [
        [1,2,3,4,5],
        [6,7,8,9,10],
        [0,1,2,3,4],
        [5,6,7,8,9]
        ]
    b = np.array(a, dtype=np.float32).reshape(4, 5)
    c = average_record(b, 2)
    print(c)


if __name__=="__main__":
    test_average_record()
