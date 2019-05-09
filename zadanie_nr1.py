#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.applications.vgg19 import VGG19

PATH_TO_IMAGES = 'common-mobile-web-app-icons'


def main(args):
    model = VGG19()
    print(model.summary())
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
