import re
import sys, os
from keras.layers.merge import concatenate
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import numpy as np
import pandas as pd
from keras import optimizers
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Model


train_file = ('train.txt')
test_file = ('test.txt')
X_test = []
X_train = []
file = open(train_file, 'r')
train_text = file.readlines()
file.close()
file = open(test_file, 'r')
test_text = file.readlines()
file.close()
seq_length = 210
protein_dict = {
                'A':1,
                'C':2,
                'D':3,
                'E':4,
                'F':5,
                'G':6,
                'H':7,
                'I':8,
                'K':9,
                'L':10,
                'M':11,
                'N':12,
                'P':13,
                'Q':14,
                'R':15,
                'S':16,
                'T':17,
                'V':18,
                'W':19,
                'Y':20}


def model_performace(test_num, pred_y, labels):
    tn, fp, fn, tp = confusion_matrix(labels, pred_y).ravel()
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = matthews_corrcoef(labels, pred_y)
    return acc, precision, sensitivity, specificity, MCC


def lab_train():
    label = []
    with open(train_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                seq_label = values[1]
                if seq_label == '1':
                    label.append(1)
                else:
                    label.append(0)
    return label


def lab_test():
    label = []
    with open(test_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                seq_label = values[1]
                if seq_label == '1':
                    label.append(1)
                else:
                    label.append(0)
    return label


def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])

    return myFasta


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def CKSAAGP(fastas, gap = 5, **kw):

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings


for i in range(len(train_text)//2):
    line = train_text[i*2+1]
    line = line[0:len(line)-1]
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[seq_length - 1 - j] = protein_dict[line[len(line) - j - 1]]
    X_train.append(seq)
X_train = np.array(X_train)


for i in range(len(test_text)//2):
    line = test_text[i*2+1]
    line = line[0:len(line)-1]
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[seq_length - 1 - j] = protein_dict[line[len(line) - j - 1]]
    X_test.append(seq)
X_test = np.array(X_test)


train_fasta = readFasta(train_file)
train_encodding_cksaagp = CKSAAGP(train_fasta)
train_encodding_cksaagp = pd.DataFrame(train_encodding_cksaagp)
train_encodding_cksaagp = np.array(train_encodding_cksaagp)


test_fasta = readFasta(test_file)
test_encodding_cksaagp = CKSAAGP(test_fasta)
test_encodding_cksaagp = pd.DataFrame(test_encodding_cksaagp)
test_encodding_cksaagp = np.array(test_encodding_cksaagp)


def DLFF_ACP(train, test, train_cksaagp, test_cksaagp):
    nbf = 32
    flen = 16
    ndrop = 0.2
    nbatch = 29
    nepochs = 8
    npooling = 8
    train_label = lab_train()
    test_label = lab_test()

    fea_input = Input(shape=(150,))
    fea_cnn3 = Dense(128, activation='relu')(fea_input)
    fea_cnn3 = Dropout(ndrop)(fea_cnn3)
    fea_cnn3 = Dense(64, activation='relu')(fea_cnn3)

    embedding_layer = Embedding(21,
                                128,
                                input_length=seq_length,
                                )
    sequence_input = Input(shape=(seq_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    cnn3 = Conv1D(filters=nbf, kernel_size=flen, padding='same', activation='relu')(embedded_sequences)
    cnn3 = MaxPooling1D(pool_size=npooling)(cnn3)
    cnn3 = Flatten()(cnn3)
    cnn3 = Dense(64, activation='relu')(cnn3)
    con = concatenate([cnn3, fea_cnn3], axis=-1)
    con = Dense(64, activation='relu')(con)
    preds = Dense(1, activation='sigmoid')(con)
    model = Model([fea_input, sequence_input], preds)
    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.fit([train_cksaagp, train], np.array(train_label), epochs=nepochs, batch_size=nbatch, verbose=1)
    preds = model.predict([test_cksaagp, test])
    pred_class = np.rint(preds)
    acc, precision, sensitivity, specificity, MCC = model_performace(len(test_label), pred_class, test_label)
    roc = roc_auc_score(test_label, preds) * 100.0
    print(acc, precision, sensitivity, specificity, MCC, roc)


DLFF_ACP(X_train, X_test, train_encodding_cksaagp[1:, 1:], test_encodding_cksaagp[1:, 1:])






