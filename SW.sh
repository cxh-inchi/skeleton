#!/bin/bash 
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
tensorboard --logdir=logs_train --port=6032