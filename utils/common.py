# -*- coding:utf-8 -*-
"""
    Description:
    Author: shelldream
    Date:
"""
import sys
reload(sys).setdefaultencoding("utf-8")

class colors:
    BLUE = '\033[01;34m'
    GREEN = '\033[01;32m'
    RED = '\033[01;31m'
    YELLOW = '\033[01;33m'
    ENDC = '\033[00m'

if __name__ == "__main__":
    print colors.BLUE + "It's blue!" + colors.ENDC
