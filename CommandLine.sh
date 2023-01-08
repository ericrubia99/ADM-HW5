#!/bin/bash
#1.What is the most popular pair of heroes (often appearing together in the comics)?
echo "Most popular pair of heroes:"
awk 'NR>1{c[$0]++} END{for (k in c)print k,","c[k]}' hero-network.csv | sort -k3 -nr -t,  > uniq-sort.txt
head -n 1 uniq-sort.txt 

#2.Number of comics per hero.
echo "Number of comics per hero:"
cut -d',' -f1 edges.csv | awk 'NR>1{c[$0]++} END{for (k in c)print k,","c[k]}' >num_com_per_hero.txt
head -n 10 num_com_per_hero.txt


#3.Average number of heroes per comic
echo "Average number of heroes per comic:"
sort -u edges.csv | cut -d',' -f1 edges.csv | awk 'NR>1{c[$0]++} END{for (k in c)print k,","c[k]}' | awk -F',' '{sum+=$2; ++n} END { print "Avg: "sum"/"n"="sum/n }' 
