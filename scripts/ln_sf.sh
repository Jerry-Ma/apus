#! /bin/sh
#
# do_dummy.sh
# Copyright (C) 2016 Jerry Ma <jerry.ma.nk@gmail.com>
#
#----------------------------------------------------------------------------
#"THE BEER-WARE LICENSE" (Revision 42):
#Jerry wrote this file. As long as you retain this notice you
#can do whatever you want with this stuff. If we meet some day, and you think
#this stuff is worth it, you can buy me a beer in return Poul-Henning Kamp
#----------------------------------------------------------------------------


readlink=$(which greadlink)
if [[ ! $readlink ]]; then
    readlink='which readlink'
fi
src=$1
shift
for tar in $@; do
    echo ln -sf $($readlink -m $src) $tar
    ln -sf $($readlink -m $src) $tar
done
