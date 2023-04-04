from tests import *
for test in [test_solve]:
    try:
        test()
    except NotImplementedError as e:
        continue