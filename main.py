from tests import *
for test in [test_euler, test_get_b, test_get_L, test_solve]:
    try:
        test()
    except NotImplementedError as e:
        continue