#!/bin/bash
cd $(dirname `ls -l $0 |awk '{print $NF;}'`)/
wkdir=`pwd`

rm -f $wkdir/temp.cache || true

gdcmd="$wkdir/vw -b 18
          --ftrl
	  --ring_size 1024
	  --cache_file temp.cache 
	  --passes 1
	  --save_per_pass
          --l1 0.0001
	  --l2 0.0001
	  --adaptive 
          --ftrl_alpha 0.1
	  -d /dev/stdin 
          -i ftrl.model 
          -f ftrl.resume
          --save_resume
          --readable_model ftrl.resume 
          --progressive_validation ftrl.evl
	  --loss_function=logistic" 

$gdcmd >vw.out 2>vw.err 


