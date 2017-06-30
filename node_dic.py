import csv
import json

def main():
    
    my_file1 = open("node.csv")
    node = []
    code_lst = []
    for row in csv.reader(my_file1):
        d = {"code":row[2],"name":row[1],"chapter":row[0]}
        code_lst.append(row[2])
        node.append(d)
    
    my_file2 = open("cos_similarity_full_DSM5_3ngram.csv")
    link = []

    k = 0
    
    for row in csv.reader(my_file2):
        k += 1 
        for i in range(len(code_lst)-k+1):
            d1 = {"source":row[0],"target":code_lst[i+k-1],"value":row[i+k]}
            link.append(d1)
    
    my_dic = {"nodes":node,"links":link}

    
    with open('node_dic_3.json', 'w') as fp1:
        json.dump(my_dic, fp1)
    
main()
