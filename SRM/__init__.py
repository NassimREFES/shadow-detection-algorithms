# -*- coding: utf-8 -*-

import numpy

from UnionFind import UnionFind

import cv2

from copy import deepcopy

from matplotlib import pyplot as plt

class SRM:
    def __init__(self, image, seeds, Q=32.0, Tv=45):
        self._height = image.shape[0] # longueur
        self._width = image.shape[1]  # largeur
        if image.ndim == 3: # si image = RGB ou autre dimention
            self._depth = image.shape[2] # 3 ou autre
        else:
            self._depth = 1 # image niveau de gris = 1
        
        self._n = self._width * self._height # taille de l'image
        n = self._n
        self._image = image.reshape(n, -1) # redefinir la taille de l'image ( chaque element dans une ligne)
        
        self._logdelta = 2.0 * numpy.log(6.0 * n)
        self._smallregion = int(0.0001 * n) # les petites region contiennent int(0.001 * taille) elements
        # params qui quantifie et controle la complexité stats { choix de convolution = 1 } => Q r.v.
        # Q r.v apartient a { 1, 2, ..., g } | en pratique g = 256
        self._q = Q 
        self._seeds = seeds
        self._seeds_result = numpy.zeros((self._height, self._width), dtype=numpy.uint8)

        self._count_merge_region = 0
        self._Tv = Tv
    
    def run(self):
        self.initialization() # initialisation des vars et seed de depart
        self.segmentation()
        return self.finalize()
    
    def initialization(self):
        print "init"
        self._uf = UnionFind() # crée un container UnionFind pour les regions
        uf = self._uf
        n = self._n # taille de l'image w * h
        height = self._height
        width = self._width
        depth = self._depth # 3eme dim
        
        img = self._image
        self._data = numpy.empty([n, depth + 2]) # taille image, 3eme dim, 2 pour position (i, j)
        self._sizes = numpy.ones(n)
        
        for i in xrange(height):
            for j in xrange(width):
                idx = i * width + j
                uf[idx] # chaque pixel est un seed de depart
                self._data[idx, 0:depth] = deepcopy(img[idx]) # valeur du pixel
                self._data[idx, depth ] = i # ligne
                self._data[idx, depth+1 ] = j # colonne

        self._xdata = deepcopy(self._data)

    def segmentation(self):
        pairs = self.pairs() # crée des pairs de pixels trié par ordre selon une f(. , .)
        print "segmentation"
        
        for (r1, r2) in pairs:
            r1 = self._uf[r1] # region du premier pixel de la pair
            r2 = self._uf[r2] # region du 2eme pixel de la pair
            
            # si les deux pixel ne sont pas dans la meme region
            # ET
            # si le predicat est satisfait entre les deux pixel
            if r1 != r2 and self.predicate(r1, r2):
                self.merge(r1, r2)
        
        self.merge_small_regions()

# --------------------------------------------------------
# ------------------- Partie 1 ---------------------------
# --------------------------------------------------------
    
    def pairs(self):
        print "pairs"
        pairs = []
        height = self._height
        width = self._width
        
        # crée des pairs entre (l'element actuel, element DROITE) | (l'element actuel, BAS)
        # using a C4-connectivity
        for i in xrange(height - 1):
            for j in xrange(width - 1):
                # if self._seeds[i][j]:
                # print '1> pairs'
                idx = i * width + j
                # left
                pairs.append( ( idx, i * width + j + 1) )

                # below
                pairs.append( ( idx, (i + 1) * width + j) )
        
        pairs = self.sort(pairs) # tri par ordre croissant
        return pairs
    
    def sort(self, pairs):
        print "sort"
        img = self._image
        def diff(p):
            (r1, r2) = p
            diff = numpy.max(numpy.abs(img[r1] - img[r2]))
            return diff
        return sorted(pairs, key=diff)

# --------------------------------------------------------
# ------------------- Partie 2 ---------------------------
# --------------------------------------------------------
    
    def predicate(self, r1, r2):
        g = 256.0 # niveau de gris en pratique
        logdelta = self._logdelta
        
        w = self._sizes # poid de chaque pixel ( par defaut = 1 )
        out = self._data # tableau de pixels de l'image
        
        d2 = (out[r1] - out[r2])**2 # difference entre pixels d'une pair
        
        log_r1 = min(g, w[r1]) * numpy.log(1.0 + w[r1])
        log_r2 = min(g, w[r2]) * numpy.log(1.0 + w[r2])
        
        q = self._q # la Qv.r. choisi 
        dev1 = g**2 / (2.0 * q * w[r1]) * (log_r1 + logdelta)
        dev2 = g**2 / (2.0 * q * w[r2]) * (log_r2 + logdelta)
        dev = dev1 + dev2
        
        # verifier que tout les valeurs dans d2 sont inferieur au seuil dev
        # d2 contient valeur R, G, B
        
        return (d2 < dev).all()
    
    def merge(self, r1, r2):
        # merge de r1 et r2 dans une seul region r
        r = self._uf.union(r1, r2)
        
        # recuperer le poid de la region r1 et r2
        s1 = self._sizes[r1]
        s2 = self._sizes[r2]
        n = s1 + s2 # somme des deux poid

        # affecter une valeur moyenne a la region
        self._data[r] = (s1 * self._data[r1] \
            + s2 * self._data[r2]) / n

        # affecter le poid a la region mergé
        self._sizes[r] = n

        self._count_merge_region += 1

        return r
    
    def merge_small_regions(self):
        height = self._height # longueur de l'image
        width = self._width   # largueur de l'image
        smallregion = self._smallregion
        
        # verifier l'appartenance des pixels adjacent
        for i in xrange(self._height):
            for j in xrange(1, self._width):
                # if self._seeds[i][j]:
                # print '2> small region'
                idx = i * width + j
                r1 = self._uf[idx]
                r2 = self._uf[idx - 1]
                
                # si r1 et r2 n ont pas la meme racine ( dans une meme region )
                # ET
                # si l'un des deux region est inferieur au seuil(petit region)
                if r1 != r2 and ( self._sizes[r1] < smallregion or self._sizes[r2] < smallregion):
                    self.merge(r1, r2)
    
    def finalize(self):
        height = self._height
        width = self._width
        depth = self._depth
        
        uf = self._uf 
        data = self._data[:, 0:depth]
        out = numpy.empty([self._n, depth])

        # xdata = self._xdata[:, 0:depth]
        # xout = numpy.empty([self._n, depth])

        result = numpy.zeros((self._height, self._width), dtype=numpy.uint8)
        classes, classes_counter = self.get_classes()

        for i in xrange(height):
            for j in xrange(width):
                idx = i * width + j
                # print '2> ', uf[idx]
                r1 = uf[idx] # region r1 a qui appartient l'element idx
                out[idx] = data[r1]
                if self._seeds[i, j]:
                    if sum(data[r1])/len(data[r1]) < self._Tv :
                        for ii, jj in classes[r1]:
                            result[ii, jj] = 255
        
        self._finalized = out.reshape(height, width, -1) # reshape de tableau a la matrice


        self._xfinalized = result

        fermeture_morph = self._xfinalized

        return self._finalized, self._xfinalized, fermeture_morph

    def get_classes(self):
        height = self._height
        width = self._width
        depth = self._depth

        classes = {}
        classes_counter = {}

        uf = self._uf
        data = self._data[:, 0:depth]
        seeds = self._seeds

        for i in xrange(height):
            for j in xrange(1, width):
                idx = i * width + j
                r1 = uf[idx]
                if r1 in classes:
                    classes[r1].append((i, j))
                    classes_counter[r1][1] += 1
                else:
                    classes[r1] = [(i, j)]
                    classes_counter[r1] = [0, 1]
                
                if seeds[i, j]:
                    classes_counter[r1][0] += 1

        return classes, classes_counter
    
    def map(self):
        # recuperer le nombre d'element dans chaque region
        height = self._height
        width = self._width
        depth = self._depth
        
        classes = {}
        
        uf = self._uf
        data = self._data[:, 0:depth]
        out = numpy.empty(self._n)
        for i in xrange(height):
            for j in xrange(1, width):
                idx = i * width + j
                r1 = uf[idx]
                if r1 in classes:
                    classes[r1] += 1
                else:
                    classes[r1] = 1
                out[idx] = r1
        
        return classes, out.reshape(height, width)
    
    def exploded(self):
        out0 = numpy.empty([n, depth])
        out = self._data
        expl = numpy.zeros([2 * self._height, 2 * self._width, self._depth])
        
        for i in xrange(self._height):
            for j in xrange(1, self._width):
                r1 = self._uf[r]
                x = int(self._more[r1, 0])
                y = int(self._more[r1, 1])
                expl[i + x, j + y] = out[r]
        
        return expl

if __name__ == "__main__":
    import sys
    from scipy.misc import imread
    from matplotlib import pyplot
    
    q = int(sys.argv[1])
    im = imread(sys.argv[2])
    
    srm = SRM(im, q)
    
    # srm.initialization()
    # srm.segmentation()
    # classes, map = srm.map()
    # print classes
    # pyplot.imshow(map)
    # pyplot.show()
    
    segmented = srm.run()
    pyplot.imshow(segmented/256)
    pyplot.show()
    
    # exploded = srm.exploded()
    # pyplot.imshow(exploded/256)
    # pyplot.show()
    
