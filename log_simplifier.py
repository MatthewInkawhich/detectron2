# MatthewInkawhich

"""
This script reads in a raw log file and outputs a simplified csv with 
the info we need.
"""

import argparse
import os
import csv


##################################
### MAIN
##################################
def main():
    # Get log path from arg
    parser = argparse.ArgumentParser(description="Log Simplifier")
    parser.add_argument(
        "logpath",
        type=str,
    )
    args = parser.parse_args()

    # Read file into list of lines
    lines = [line.rstrip('\n') for line in open(args.logpath)]
    # Filter down to lines of interest
    output_lists = []
    for i in range(len(lines)):
        split_line = lines[i].split()
        if len(split_line) >= 2:
            # Add AP score rows
            if split_line[0] == "[WORST]" or split_line[0] == "[MEDIAN]" or split_line[0] == "[BEST]":
                s = []
                version = split_line[0].split('[')[1].split(']')[0]
                s.append(version)
                aps = lines[i+3].split("copypaste:")[1].split(",")
                s.extend(aps)
                output_lists.append(s)

                
            #if split_line[1] == "*":
            #    output_lists.append([split_line[3], split_line[5], split_line[7]])
            #if split_line[0] == "Choice" and split_line[1] == "Counts":
            #    output_lists.append([split_line[2]])
            #if split_line[1] == "Block:":
            #    s = lines[i].split('tensor([')[1].split('])')[0].split(',')
            #    if "])" not in split_line[-1]:
            #        s2 = lines[i+1].split('])')[0].split(',')
            #        s.extend(s2)
            #    s.insert(0, split_line[2])
            #    s = [p.strip() for p in s]
            #    s = [p for p in s if p]
            #    output_lists.append(s)
            if split_line[0] == "DONE:":
                output_lists.append([split_line[1]])
                output_lists.append([])
                output_lists.append([])
            

    #for l in output_lists:
    #    print(l)

    # Write to file
    with open('tmp.csv', mode='w') as tmpfile:
        tmp_writer = csv.writer(tmpfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(output_lists)):
            print(output_lists[i])
            tmp_writer.writerow(output_lists[i])
    





if __name__ == "__main__":
    main()
