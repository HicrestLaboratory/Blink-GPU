#!/bin/bash
recipepath="csvs2merge.txt"

while read p
do 
	echo "$p"
	$p
done <"${recipepath}"
