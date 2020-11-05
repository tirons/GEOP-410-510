import numpy as np 
import matplotlib.pyplot as plt
import sys, os

def writeRecord(key, record, job):
    Hz = record[3].split()[0]
    fdir = "./ZeroTEM/"+job[1]+"/"+job[3]+job[4]+"/"+Hz+"Hz"
    if not os.path.isdir(fdir):
        os.mkdir(fdir)

    fout = open(fdir+"/"+str(key)+".SND", 'w')
    for line in record:
        fout.write(line)
    fout.close()

def parseCAP(filename):
    inp = open(filename, 'r')

    ls = 0
    startLine = []
    iline = 0

    lines = inp.readlines()

    if not os.path.isdir("./ZeroTEM"):
        os.mkdir('ZeroTEM')

    for line in lines:
        if len(line) == 5:
            startLine.append(iline)    
        iline += 1    

    records = {}
    # Split the records  
    for number, sline, in enumerate(startLine[0:-1]):
        records[number] = lines[sline:startLine[number+1]]
    # last record
    records[number+1] = lines[startLine[-1]:len(lines)]

    print("##########################################")
    print(str(len(records)) + " records are being parsed  ")
    print("##########################################")

    for key in records:
        #print (records[key][1].split())
        #exit()
        if records[key][3][0:3] == "JOB":
            job = records[key][3].split()
            if not os.path.isdir("./ZeroTEM/"+job[1]):
                os.mkdir("./ZeroTEM/"+job[1])
            if not os.path.isdir("./ZeroTEM/"+job[1]+"/"+job[3]+job[4]):
                os.mkdir("./ZeroTEM/"+job[1]+"/"+job[3]+job[4])
        else: 
            writeRecord(key, records[key],job)

    

if __name__ == '__main__':
    parseCAP(sys.argv[1])
