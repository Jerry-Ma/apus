#! /bin/sh
#
# ldac_to_asc.sh
# Copyright (C) 2016 Jerry Ma <jerry.ma.nk@gmail.com>
#
#----------------------------------------------------------------------------
#"THE BEER-WARE LICENSE" (Revision 42):
#Jerry wrote this file. As long as you retain this notice you
#can do whatever you want with this stuff. If we meet some day, and you think
#this stuff is worth it, you can buy me a beer in return Poul-Henning Kamp
#----------------------------------------------------------------------------


if [[ ! $2 ]]; then
    exit 1
fi
ldactoasc $1 > $2
