#!/usr/bin/env python

# ----------------------------------------------------------------------------------- #
#  Simple stopwatch macro, writing result into a file (to be read by analysis program)
#  Python 3 version
#
#  Author: Troels C. Petersen (Niels Bohr Institute)
#  Date:   20-11-2020 (last update)
#
# ----------------------------------------------------------------------------------- #

from time import time
import os,sys

name = "timer_output.dat"

n = 0
while (os.path.isfile(name)):
    print("The file ",name," already exists in this directory.")
    name = input("Please enter a new name for your output file: ")
    
newname = name
    
# Make some check about the string entered
if len(newname)==0:
    good_name=False
else:
    good_name=(newname[0].isalpha())*(newname.endswith(".dat"))
    
while n<5 and (not good_name):
    newname = input("Name must end with .dat and first character must be a letter: ")
    n+=1
    if len(newname)==0:
        newname = input("Name must not be empty: ")
        n+=1
        continue
    else:
        good_name=(newname[0].isalpha())*(newname.endswith(".dat"))
            
if not good_name:
    sys.exit("ERROR: Filename does not fulfill basic requirements.")
else:
    name = newname
        
    
with open(name, "w") as outfile : 
    now = time()
    laptime = 0.0
    counter = 0
    while( input( "%4d \t %10.4f \t Laptime by enter, Exit by key+enter \t"%(counter, laptime) ) is "" ) : 
        counter += 1
        laptime = time()-now
        outfile.write("%4d \t %10.4f \n"%(counter, laptime))
        
    print("Done.")
