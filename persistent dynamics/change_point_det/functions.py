import numpy as np
import gudhi
from scipy.sparse import csr_matrix
from gtda.time_series import SingleTakensEmbedding
from gtda.graphs import GraphGeodesicDistance
from gtda.homology import FlagserPersistence

def bottleneck0(X,Y):
    """
    Returns the dimension 0 bottleneck distance between two non-empty persistence diagrams.
    All components are assumed to be born at the beginning of the filtration, 
    that is, all non-trivial points lie in the vertical axis.   
    
    Parameters
    ----------
    X : 1D numpy array
        Non-empty array of death times for persistence diagram 1.
    Y : 1D numpy array
        Non-empty array of death times for persistence diagram 2.
           
    Returns
    -------
    d : float
        The bottleneck distance between X and Y.
    """
    
    #Swap if X is longer
    if len(Y) < len(X):
        X_copy = np.copy(X)
        X = Y
        Y = X_copy  
    
    #Initialize
    X=-np.sort(-X) 
    Y=-np.sort(-Y)
    d=0
    N = len(X)

    Z = abs(X-Y[0:N]) 
    l = np.argmax(Z) 
    dtemp = Z[l]

    # Maintain an initial bijection. Systematically modify this bijection to optimize
    # the norm between matched points until the bottleneck matching is recovered.
    if N != len(Y) and dtemp < 0.5*Y[N]:
        d = 0.5*Y[N]      
    elif l >= 0 and len(Z) > 1:
        while len(Z) > 1:
            k = 0.5*max(X[l],Y[l])
            if np.max(np.delete(Z,l)) <  k and k < dtemp: 
                d = k
                break

            elif np.max(np.delete(Z,l)) >= k:
                if len(Z[Z>=k]) == len(np.where(np.where(Z >= k)[0] >= l)[0]): 
                    d = k
                    break
                else:
                    Z=Z[0:l]
                    X=X[0:l]
                    Y=Y[0:l]
                    l=np.argmax(Z)
                    dtemp = Z[l]             
                    if len(Z) == 1:
                        d = min(dtemp,0.5*max(X[l],Y[l]))
                        break
            else:
                d = dtemp
                break
    else:
        # If one of the persistence diagrams is a singleton 
        d = min(dtemp,0.5*max(X[0],Y[0]))

    return d

def persistence(time_series, emdim, tilay, homdim = [0]):
    '''
    Berechnung des Persistenz Diagrams einer Zeitreihe. Die Zeitreihe wird im
    R^n eingebettet, die Vektoren geordnet und einer Permutation in der Sn
    zugeordnet. Dann wird ein Graph gebildet in dem die Ecken die Permutationen
    sind und Kanten existieren wenn eine Permutation auf die andere folgt. Von
    dem Graph wird dann schließlich die Persistente Homologie berechnet.
    '''

    STE = SingleTakensEmbedding(parameters_type='fixed',
                                time_delay=tilay,dimension=emdim)
    ts_embedded = STE.fit_transform(time_series)

    ps = np.argsort(ts_embedded)

    u = np.unique(ps, axis=0, return_inverse=True)
    x_axis = [x for x in range(0,len(ps))]
    y_axis = u[1]

    adj = [[0 for permutation in u[0]]for permutation in u[0]]

    for x in range(0,len(u[1])-1):
        a = u[1][x]
        b = u[1][x+1]

        if a != b:
            adj[a][b] += 1

    max_entry = 0

    for i in adj:
        for j in i:
            if j >= max_entry:
                max_entry = j

    n = len(u[0])
    row = []
    col = []
    data = []

    adj = [[0 for permutation in u[0]]for permutation in u[0]]

    for x in range(0,len(u[1])-1):
        a = u[1][x]
        b = u[1][x+1]

        if a != b:
            if adj[a][b] == 0:
                adj[a][b] += 1
                row.append(a)
                col.append(b)
                data.append(max_entry)

            else:
                row.append(a)
                col.append(b)
                data.append(-1)

    transition_graph = csr_matrix((data, (row,col)), shape=(n,n))
    '''
    In dem Transistion Graph hat die am meisten (n mal) durchlaufene Ecke
    Länge 1 und die anderen (k mal durchlaufen) Länge n-k+1.
    Der Graph ist gerichtet.
    '''

    ggd = GraphGeodesicDistance(directed=True)
    dm = ggd.fit_transform([transition_graph])

    fp = FlagserPersistence(homology_dimensions = homdim, n_jobs=-1)
    hom = fp.fit_transform(dm)
    #fp.plot(hom).show()
    
    return hom

def bottleneck0_transformer(hom):
    homnull = []
    for d in hom:
        for c in d:
            if c[2]==0:
                y = c[1]
                homnull.append(y)

    homnull = np.array(homnull)
    return homnull