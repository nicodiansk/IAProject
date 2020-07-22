import numpy as np, scipy as sp, time, scipy.io
import aknnminecopy

if __name__ == '__main__':
    '''
    Importazione del dataset notMNIST e conversione da file .mat a .py seguendo le istruzioni date dal creatore del dataset.
    '''
    print("Importo dataset...")
    notMNIST_small = scipy.io.loadmat("notMNIST_small.mat")['images'].reshape(784, 18724)
    print(notMNIST_small.shape, type(notMNIST_small))
    print("Dataset importato correttamente.")
    nmn = (notMNIST_small.T - 255.0 / 2) / 255.0   #resize
    labels = scipy.io.loadmat("notMNIST_small.mat")['labels'].astype(int)
    labels_to_symbols = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
    labels = np.array([labels_to_symbols[x] for x in labels])
    print(labels.shape)
    # Calcola la lista dei k esatti nearest neighbors Euclidei per ogni punto usando l'apposita funzione in aknn_alg.py.
    itime = time.time()
    nbrs_list = aknnminecopy.calc_nbrs_exact(nmn, k=1000)
    print('Neighbor indices computed. Time:\t {}'.format(time.time() - itime))

    itime = time.time()
    aknn_predictions = aknnminecopy.predict_nn_rule(nbrs_list, labels)
    print('AKNN predictions made. Time:\t {}'.format(time.time() - itime))

    #Confronto tra kNN e AkNN.
    kvals = [3, 5, 7, 8, 10, 30, 100]
    for i in range(len(kvals)):
        knn_predictions = aknnminecopy.knn_rule(nbrs_list, labels, k=kvals[i])
        aknn_cov_ndces = aknn_predictions[1] <= kvals[i]
        aknn_cov = np.mean(aknn_cov_ndces)
        aknn_condacc = np.mean((aknn_predictions[0] == labels)[aknn_cov_ndces])
        print('{}-NN accuracy: \t\t{}'.format(kvals[i], np.mean(knn_predictions == labels)))
        print('AKNN accuracy (k <= {}): \t{} \n Coverage: \t{}\n'.format(
            kvals[i], aknn_condacc, aknn_cov))
    print('Overall AKNN accuracy: {}'.format(np.mean(aknn_predictions[0] == labels)))

