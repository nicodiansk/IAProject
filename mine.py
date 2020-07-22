import numpy as np, scipy as sp, time, scipy.io
from PIL import Image
from cv2 import *
import os
import aknnminecopy


def load_images_from_folder(folder, images):

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), 0) #greyscale
        img=cv2.resize(img,(100,100), interpolation=INTER_AREA)

        '''#Per vedere la correttezza delle immagini ridotte
        cv2.imshow('img', img)
        cv2.waitKey()'''

        if img is not None:
            images.append(img)
    return images

def load_labels(folder, labels_db):
    for filename in os.listdir(folder):
        if filename is not None:
            label=filename[:2]
            labels_db.append(label)
    return labels_db
if __name__ == '__main__':
    '''
    Importazione dataset con immagini che sono numpy arrays
    '''
    print("Importo dataset...")
    images = []
    labels_db = []

    images=load_images_from_folder('./training/training/', images)

    labels_db=load_labels('./training/training/', labels_db)
    print("Dataset importato correttamente.")
    np_images=np.asarray(images).reshape(10000, 1098)

    np_labels=np.asarray(labels_db)
    print(type(np_images),type(np_labels))
    print(np_images.shape, np_labels.shape)
    print('Conversione effettuata')


    #notMNIST_small = scipy.io.loadmat("notMNIST_small.mat")['images'] #.reshape(784, 18724)
    #print(notMNIST_small.shape)
    nmn = (np_images.T - 255.0 / 2) / 255.0   #resize
    #nmn = np_images.T

    # Calcola la lista dei k esatti nearest neighbors Euclidei per ogni punto usando l'apposita funzione in aknn_alg.py.
    itime = time.time()
    print('Inizio calcolo...')
    nbrs_list = aknnminecopy.calc_nbrs_exact(nmn, k=1000)
    print('nbrs shape', nbrs_list.shape)
    print(nbrs_list)
    print('Neighbor indices computed. Time:\t {}'.format(time.time() - itime))

    itime = time.time()
    aknn_predictions = aknnminecopy.predict_nn_rule(nbrs_list, np_labels)
    print('AKNN predictions made. Time:\t {}'.format(time.time() - itime))

    #Confronto tra kNN e AkNN.
    kvals = [3, 5, 7, 8, 10, 30, 100]
    for i in range(len(kvals)):
        knn_predictions = aknnminecopy.knn_rule(nbrs_list, np_labels, k=kvals[i])
        aknn_cov_ndces = aknn_predictions[1] <= kvals[i]
        aknn_cov = np.mean(aknn_cov_ndces)
        aknn_condacc = np.mean((aknn_predictions[0] == np_labels)[aknn_cov_ndces])
        print('{}-NN accuracy: \t\t{}'.format(kvals[i], np.mean(knn_predictions == np_labels)))
        print('AKNN accuracy (k <= {}): \t{} \n Coverage: \t{}\n'.format(
            kvals[i], aknn_condacc, aknn_cov))
    print('Overall AKNN accuracy: {}'.format(np.mean(aknn_predictions[0] == np_labels)))


