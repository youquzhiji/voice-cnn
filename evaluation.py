import csv
#Transform meta to a dict of labels
gt={}
with open("vox1_meta.csv", "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in reader:
        # results.append([line[0],line[2]])
        gt[line[0]]=line[2]

# with open('vox1_label.csv', 'w', newline='') as myfile:
#     wr = csv.writer(myfile, delimiter="\t",quoting=csv.QUOTE_ALL)
#     wr.writerows(results)

# gt={}
# with open("vox1_label.csv", "r") as f:
#     reader = csv.reader(f, delimiter="\t")
#     for line in reader:
#         # gt[line[0]]='female' if line[1]=='f' else 'male'
#         gt[line[0]]=line[1]

tp=tn=fp=fn=0
with open("myseg.csv", "r") as f: # Open csv file with pred results
    reader = csv.reader(f, delimiter="\t")
    next(reader,None)
    for line in reader:
        if line[1]=='female' and gt[line[0]]=='f':
            tp+=1
        elif line[1]=='female' and gt[line[0]]=='m':
            fp+=1
        elif line[1]=='male' and gt[line[0]]=='f':
            fn+=1
        else:
            tn+=1
print('female samples in ground truth: {}'.format(tp+fn))
print('male samples in ground truth: {}'.format(tn+fp))
print('accuracy: {}'.format((tp+tn)/(tp+tn+fp+fn)))
print('precision for female: {}'.format(tp/(tp+fp)))
print('precision for male: {}'.format(tn/(tn+fn)))
print('recall for female: {}'.format((tp)/(tp+fn)))
print('recall for male: {}'.format((tn)/(tn+fp)))