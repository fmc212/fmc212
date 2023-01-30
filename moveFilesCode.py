# move files code 

import numpy as np 
import pandas as pd 
import shutil

list_path = "C:/Users/fcarter/Desktop/ComponentList.csv"
abs_path = "C:/Users/fcarter/DMG MORI/U.S. SLM Development Project - General/00_TECHNICAL/09_Prototype_Production/Models for Quote"



df = pd.read_csv(list_path, engine="python")

bsz = df.values[:,0]
bsz1 = bsz[~np.isnan(bsz)]

am1 = df.values[:,1]
am11 = am1[~np.isnan(am1)]

amu = df.values[:,2]
amu1 = amu[~np.isnan(amu)]

ah1 = df.values[:,3]
ah11 = ah1[~np.isnan(ah1)]


def moveFiles(fileArray, assemblyCode): 

    for i in range(len(fileArray)): 
        
        fileNum = int(fileArray[i])
        fileNum = f'{fileNum}'

        if abs_path + fileNum + '.stp': 
            
            file_name = '/' + fileNum + '.stp'
            dst_folder = abs_path + '/' + assemblyCode

            #print(file_name)
            #print(dst_folder)

            #print(abs_path + file_name)
            #print(dst_folder + file_name)
            
            try: 
                shutil.move(abs_path + file_name, dst_folder + file_name)
            except: 
                print(file_name + "DNE")
        else: 
            print("File Does Not Exist")


if __name__ == "__main__": 
    moveFiles(bsz1, "BSZ")
    moveFiles(am11, "AM1")
    moveFiles(amu1, "AMU")
    moveFiles(ah11, "AH1")