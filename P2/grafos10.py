import sys
import time
import os
import random
import numpy as np
import pickle
import gzip
import queue as qe
import time
import networkx as nx

"""
Practica 1

Adrian Navas Ajenjo
Gloria del Valle Cano

Pareja 10
"""

def rand_matr_pos_graph(n_nodes, sparse_factor, max_weight=50., decimals=0):
    """
    Funcion que genera grafos de manera aleatoria.
    Devuelve la matriz de adyacencia de un grafo dirigido ponderado.

    	n_nodes: numero de nodos
    	sparse_factor: proporcion de ramas
    	max_weight: peso maximo
    	decimals: numero de decimales
    """
    l = np.full((n_nodes,n_nodes),np.inf)
    branch_number = int(sparse_factor*(n_nodes)*(n_nodes-1))
    for count in range(0,branch_number):
        condition = True
        while(condition):
            i = random.randint(0,n_nodes-1)
            j = random.randint(0,n_nodes-1)
            if(j!=i and l[i,j] == np.inf):
                condition = False
        weight = random.randint(1,max_weight)
        l[i,j] = weight
    return l

def cuenta_ramas(m_g):
    """
    Funcion que cuenta las ramas de un grafo.
    Devuelve el numero de ramas.

    	m_g: matriz de adyacencia del grafo
    """
    n_nodes = m_g.shape[0]
    branches = 0
    for i in range(0,n_nodes):
        for j in range(0,n_nodes):
            if ((m_g[i,j] != np.inf) and i != j):
                branches+=1
    return branches

def m_g_sparse_factor(m_g):
    """
    Funcion auxiliar de check_sparse_factor.
    Devuelve el calculo de sparse factor de cada grafo.

    	m_g: matriz de adyacencia del grafo
    """
    n_nodes = m_g.shape[0]
    return cuenta_ramas(m_g)/(n_nodes*(n_nodes-1))

def check_sparse_factor(n_grafos,n_nodes,sparse_factor):
    """
    Funcion que genera las matrices de un numero de grafos aleatorios con un numero de nodos y un sparse factor.
    Devuelve la media de los sparse factor reales de las matrices de los grafos generadas.

    	n_grafos: numero de grafos
    	n_nodes: numero de nodos del grafo
    	sparse_factor: proporcion de ramas
    """
    sparses = []
    for i in range(0,n_grafos):
        m = rand_matr_pos_graph(n_nodes,sparse_factor)
        sparses.append(m_g_sparse_factor(m))
    return np.mean(sparses)

def m_g_2_d_g(m_g):
    """
    Funcion que convierte la matriz de adyacencia del grafo en un diccionario.
    Devuelve un diccionario con cada nodo (k) asignado a otro diccionario (v) con los nodos destino (k)
    y el peso relacionado (v).

    	m_g: matriz de adyacencia del grafo
    """
    n_nodes = m_g.shape[0]
    d_g = {}
    for i in range(0,n_nodes):
        d_g[i] = {}
        for j in range(0,n_nodes):
            if(m_g[i,j] != np.inf):
                d_g[i][j] = m_g[i,j]
    return d_g

def d_g_2_m_g(d_g):
    """
    Funcion que convierte el diccionario del grafo a una matriz de adyacencia.
    Devuelve la matriz de adyacencia resultante.

    	d_g (diccionario) diccionario del grafo
    """
    n_nodes = len(d_g.keys())
    m_g = np.full((n_nodes,n_nodes),np.inf)
    for node in d_g:
        for branch in d_g[node]:
            m_g[node,branch] = d_g[node][branch]
    return m_g

def save_object(obj, f_name="obj.pklz", save_path="."):
    """
    Funcion que guarda un objeto en un dump file.
    Realiza el volcado en cuestion, en este caso, de un grafo.

    	obj: objeto a guardar
    	f_name: nombre del objeto
    	save_path: ruta del archivo
    """
    f = gzip.open(save_path+"/"+f_name,"wb")
    pickle.dump(obj,f)
    f.close()

def read_object(f_name, save_path="."):
    """
    Funcion que lee un objeto.
    Devuelve el objeto en cuestion.

    	f_name: nombre del archivo
    	save_path: ruta del fichero
    """
    f = gzip.open(save_path+"/"+f_name,"rb")
    obj = pickle.load(f)
    f.close()
    return obj

def d_g_2_TGF(d_g,f_name):
    """
    Funcion que pasa un grafo en forma de diccionario a formato TGF.

    	d_g (diccionario) diccionario
    	f_name: nombre del fichero donde guardamos el grafo
    """
    f = open(f_name,"w")
    for node in d_g.keys():
        f.write("%d\n" % node)
    f.write("#\n")
    for node in d_g.keys():
        for branch in d_g[node]:
            f.write("{0} {1} {2}\n".format(node, branch, d_g[node][branch]))
    f.close()

def TGF_2_d_g(f_name):
    """
    Funcion que pasa un grafo en formato TGF a un diccionario.

    	f_name: nombre del fichero donde se guardo el grafo
    """
    f = open(f_name,"r")
    line = f.read().split('\n')
    d_g = {}
    aux = []
    for c in line:
        aux.append(c)
        if(c == "#"):
            break
        n = int(c)
        d_g.update({n:{}})

    line = line[len(aux):-1]
    for c in line:
        splitted = c.split(" ")
        p1 = int(splitted[1])
        p2 = float(splitted[2])
        d_g[int(splitted[0])].update({p1:p2})
    f.close()
    return d_g

def dijkstra_d(d_g, u):
    """
    Funcion que aplica el algoritmo de Dijkstra a un grafo en formato de diccionario a partir de un nodo inicial.
    Devuelve un diccionario con las distancias mínimas al resto de nodos y otro que contiene el padre correspondiente
    a cada vértice accesible.

    	d_g (diccionario) diccionario
    	u: nodo inicial
    """
    d_dist = {}
    d_prev = {}

    dist = np.full(len(d_g.keys()), np.inf)
    visitados = np.full(len(d_g.keys()), False)
    n_padre = np.full(len(d_g.keys()), None)

    cola = qe.PriorityQueue()
    dist[u] = 0.0
    cola.put((0.0, u))

    while not cola.empty():
        pos = cola.get()
        visitados[pos[1]] = True

        for key,value in d_g[pos[1]].items():
            if dist[key] > dist[pos[1]] + value:
                dist[key] = dist[pos[1]] + value
                n_padre[key] = pos[1]
                cola.put((dist[key], key))

    for n in range(len(dist)):
        d_dist.update({n:dist[n]})
        d_prev.update({n:n_padre[n]})

    return d_dist, d_prev

def dijkstra_m(m_g,u):
    """
    Funcion que aplica el algoritmo de Dijkstra a un grafo en formato de matriz de adyacencia a partir de un nodo inicial.
    Devuelve un diccionario con las distancias mínimas al resto de nodos y otro que contiene el padre correspondiente
    a cada vértice accesible.

    	m_g: matriz de adyacencia
    	u: nodo inicial
    """
    d_dist = {}
    d_prev = {}

    n_nodos = m_g.shape[0]
    dist = np.full(n_nodos, np.inf)
    visitados = np.full(n_nodos, False)
    n_padre = np.full(n_nodos, None)

    cola = qe.PriorityQueue()
    dist[u] = 0.0
    cola.put((0.0, u))

    while not cola.empty():
        pos = cola.get()
        nodo = pos[1]
        visitados[nodo] = True

        for adyacente in range(0,n_nodos):
            if(m_g[nodo,adyacente] != np.inf):
                if dist[adyacente] > dist[nodo] + m_g[nodo,adyacente]:
                    dist[adyacente] = dist[nodo] + m_g[nodo,adyacente]
                    n_padre[adyacente] = nodo
                    cola.put((dist[adyacente], adyacente))

    for n in range(len(dist)):
        d_dist.update({n:dist[n]})
        d_prev.update({n:n_padre[n]})

    return d_dist, d_prev

def min_paths(d_prev):
    """
    Funcion que devuelve el diccionario con el camino minimo desde el nodo inicial a otro nodo

    	d_prev: diccionario que contiene el padre correspondiente a cada vértice accesible.
    """
    inicial = -1
    for key in d_prev:
        if(d_prev[key] == None):
            inicial = key
    if(inicial == -1):
        raise Exception("La lista de padres no contiene un nodo inicial con padre = None")
    d_paths = {}
    for nodo in d_prev:
        path = []
        condicion = True

        nodo_padre = nodo
        while(condicion):
            if(nodo_padre == None):
                path.reverse()
                d_paths[nodo] = path
                condicion = False
            else:
                path.append(nodo_padre)
                nodo_padre = d_prev[nodo_padre]
    return d_paths

def time_dijkstra_m(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """
    Funcion que calcula el tiempo de aplicar Dijkstra a un grafo dado en formato de matriz de adyacencia.
    Devuelve una lista de tiempos para cada grafo al que se le ha aplicado el algoritmo.

    	n_graphs: numero de grafos a generar
    	n_nodes_ini: num de nodos inicial
    	n_nodes_fin: num de nodos final
    	step: incremento
    	sparse_factor: factor proporcion de ramas
    """
    diccionario_grafos = {}
    lista_tiempos = []
    for i in range(0,n_graphs):
        for n_nodos in range(n_nodes_ini,n_nodes_fin+1,step):
            if(n_nodos not in diccionario_grafos.keys()):
                diccionario_grafos[n_nodos] = []
            m_g = rand_matr_pos_graph(n_nodos, sparse_factor)
            diccionario_grafos[n_nodos].append(m_g)

    for n_nodos in diccionario_grafos.keys():

        tiempo_ini = time.time()
        for i in range(0,n_graphs):
            for nodo in range(0,n_nodos):
                dijkstra_m(diccionario_grafos[n_nodos][i],nodo)
        tiempo_fin = time.time()-tiempo_ini
        lista_tiempos.append(tiempo_fin/n_nodos)

    return lista_tiempos

def time_dijkstra_d(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """
    Funcion que calcula el tiempo de aplicar Dijkstra a un grafo dado en formato diccionario.
    Devuelve una lista de tiempos para cada grafo al que se le ha aplicado el algoritmo.

    	n_graphs: numero de grafos a generar
    	n_nodes_ini: num de nodos inicial
    	n_nodes_fin: num de nodos final
    	step: incremento
    	sparse_factor: factor proporcion de ramas
    """
    diccionario_grafos = {}
    lista_tiempos = []
    for i in range(0,n_graphs):
        for n_nodos in range(n_nodes_ini,n_nodes_fin+1,step):
            if(n_nodos not in diccionario_grafos.keys()):
                diccionario_grafos[n_nodos] = []
            d_g = m_g_2_d_g(rand_matr_pos_graph(n_nodos, sparse_factor))
            diccionario_grafos[n_nodos].append(d_g)

    for n_nodos in diccionario_grafos.keys():

        tiempo_ini = time.time()
        for i in range(0,n_graphs):
            for nodo in range(0,n_nodos):
                dijkstra_d(diccionario_grafos[n_nodos][i],nodo)
        tiempo_fin = time.time()-tiempo_ini
        lista_tiempos.append(tiempo_fin/n_nodos)

    return lista_tiempos

def d_g_2_nx_g(d_g):
    """
    Funcion que pasa un grafo en formato de diccionario a otro de Networkx.

    	d_g (diccionario) diccionario
    """
    l_e = []
    g = nx.DiGraph()

    for key,value in d_g.items():
        for key2,value2 in value.items():
            l_e.append((key,key2,value2))

    g.add_weighted_edges_from(l_e)
    return g

def nx_g_2_d_g(nx_g):
    """
    Funcion que pasa un grafo en formato Networkx a otro en formato diccionario.

    	nx_g: grafo en formato Networkx
    """
    d_g = {}

    for count in nx_g.nodes():
        for key,value in nx_g[count].items():
            if d_g.get(count) == None:
                d_g.update({count:{}})
            for key2,value2 in value.items():
                d_g[count][key]=value2

    return d_g

def time_dijkstra_nx(n_graphs, n_nodes_ini, n_nodes_fin, step, sparse_factor=.25):
    """
    Funcion que calcula el tiempo de aplicar Dijkstra a un grafo dado en formato de Networkx.
    Devuelve una lista de tiempos para cada grafo al que se le ha aplicado el algoritmo.

    	n_graphs: numero de grafos a generar
    	n_nodes_ini: num de nodos inicial
    	n_nodes_fin: num de nodos final
    	step: incremento
    	sparse_factor: factor proporcion de ramas
    """
    diccionario_grafos = {}
    lista_tiempos = []
    for i in range(0,n_graphs):
        for n_nodos in range(n_nodes_ini,n_nodes_fin+1,step):
            if(n_nodos not in diccionario_grafos.keys()):
                diccionario_grafos[n_nodos] = []
            m_g = rand_matr_pos_graph(n_nodos, sparse_factor)
            g = nx.from_numpy_matrix(m_g)
            diccionario_grafos[n_nodos].append(g)

    for n_nodos in diccionario_grafos.keys():

        tiempo_ini = time.time()
        for i in range(0,n_graphs):
            for nodo in range(0,n_nodos):
                nx.single_source_dijkstra(diccionario_grafos[n_nodos][i],nodo)
        tiempo_fin = time.time()-tiempo_ini
        lista_tiempos.append(tiempo_fin/n_nodos)

    return lista_tiempos

def fit_plot(l, func_2_fit, size_ini, size_fin, step,legend):
    """
    Funcion que entrena un modelo lineal con unas listas de tiempos y devuelve el real y la prediccion de cada uno.

    	l: lista
    	func_2_fit: funcion en concreto a entrenar, sera en nuestro caso n**2log(n)
    	size_ini: tamaño inicial
    	size_fin: tamaño final
    	step: incremento
    	legend: parametro que hemos añadido para diferenciar las leyendas segun el formato en el que se pase el grafo
    """
    l_func_values =[i*func_2_fit(i) for i in range(size_ini, size_fin+1, step)]

    lr_m = LinearRegression()
    X = np.array(l_func_values).reshape( len(l_func_values), -1 )
    lr_m.fit(X, l)
    y_pred = lr_m.predict(X)


    trueLegend="True " + legend
    predictedLegend = "Pred. " + legend

    plt.plot(l, '*',label=trueLegend)
    plt.plot(y_pred, '-',label=predictedLegend)
    plt.legend()
    plt.title("Time comparative")
    plt.xlabel("Number of nodes")
    plt.ylabel("Time*number of nodes (seconds)")

def n2_log_n(n):
    return n**2. * np.log(n)




# Inicio P2
def graph_2_multigraph(d_g):
    """
    Funcion que crea a partir de una matriz de adyacencia, la matriz de adyacencia para un multigrafo donde
    puede existir mas de una rama entre dos nodos.

    	d_g (diccionario) lista de adyacencia del grafo a convertir
    Returns:
	d_mg: lista de adyacencia del multigrafo con las mismas ramas que d_g
    """
    d_mg = {}
    for nodo in d_g.keys():
        d_mg.update({nodo:{}})
        for k,v in d_g[nodo].items():
            d_mg[nodo].update({k:{0:v}})
    return d_mg

def rand_weighted_multigraph(n_nodes, prob=0.2, num_max_multiple_edges=1, max_weight=50., decimals=0, fl_unweighted=False,
fl_diag=True):
    """
    Funcion que crea un multigrafo dirigido con pesos aleatorio en funcion a los parametros de entrada

    	n_nodes: numero de nodos del grafo
    	prob: probabilidad de que dos nodos tengan ramas entre si
    	num_max_multiple_edges: numero maximo de ramas entre dos nodos
    	max_weight: maximo peso de una rama
    	decimals: numero de decimales del peso
    	fl_unweighted: si es True, el grafo se genera sin pesos (con peso 1 si existe rama)
    	fl_diag: True si permite auto ramas y False si no
    Returns:
	d_mg: lista de adyacencia del multigrafo creado
    """
    d_mg = {}

    for nodo in range(0,n_nodes):
        d_mg[nodo] = {}
        for rama in range(0,n_nodes):
            if( not (fl_diag == False and rama == nodo)):
                ramas = np.random.rand()
                if(ramas<prob): # si es menor, hay ramas
                    n_ramas = np.random.randint(1,num_max_multiple_edges+1)
                    for indice in range(0,n_ramas):
                        if(fl_unweighted == False):
                            peso = round(np.random.rand()*max_weight,decimals)
                        else:
                            peso = 1
                        if(rama not in d_mg[nodo].keys()):
                            d_mg[nodo][rama] = {}
                        d_mg[nodo][rama][indice] = peso
    return d_mg

def rand_weighted_undirected_multigraph(n_nodes, prob=0.2, num_max_multiple_edges=1, max_weight=50., decimals=0, fl_unweighted
=False, fl_diag=True):
    """
    Funcion que crea un multigrafo no dirigido con pesos aleatorio en funcion a los parametros de entrada

    	n_nodes: numero de nodos del grafo
    	prob: probabilidad de que dos nodos tengan ramas entre si
    	num_max_multiple_edges: numero maximo de ramas entre dos nodos
    	max_weight: maximo peso de una rama
    	decimals:  numero de decimales del peso
    	fl_unweighted: si es True, el grafo se genera sin pesos (con peso 1 si existe rama)
    	fl_diag: True si permite auto ramas y False si no
    Returns:
	d_mg: lista de adyacencia del multigrafo creado
    """
    d_mg = {}

    # creamos el grafo dirigido
    for nodo in range(0,n_nodes):
        d_mg[nodo] = {}
        #podriamos decidir si hay autorramas
        for rama in range(0,nodo):
            if( not (fl_diag == False and rama == nodo)):
                ramas = np.random.rand()
                if(ramas<prob): # si es menor, hay ramas
                    n_ramas = np.random.randint(1,num_max_multiple_edges+1)
                    for indice in range(0,n_ramas):
                        if(fl_unweighted == False):
                            peso = round(np.random.rand()*max_weight,decimals)
                        else:
                            peso = 1
                        if(rama not in d_mg[nodo].keys()):
                            d_mg[nodo][rama] = {}
                        if(nodo not in d_mg[rama].keys()):
                            d_mg[rama][nodo] = {}
                        d_mg[nodo][rama][indice] = peso
                        d_mg[rama][nodo][indice] = peso
    return d_mg

def o_a_tables(u, d_g, p, s, o, a, c):
    """
    Funcion que calcula las tablas de orden y ascenso de un grafo

    	u: nodo en el que comenzamos
    	d_g (diccionario) lista de adyacencia del grafo a estudiar
    	p: diccionario de padres
    	s: diccionario de nodos visitados
    	o, a: valores de orden y ascensos en cada nodo
    	c: orden actual
    Returns:
	c: orden actual
    """

    s[u] = True; o[u] = c; a[u] = o[u]; c += 1
    for w in d_g[u]:
        if s[w] == True and w != p[u] and o[w] < a[u]:
            a[u] = o[w]
    for w in d_g[u]:
        if s[w] == False:
            p[w] = u
            c = o_a_tables(w, d_g,p,s,o,a,c)
    for w in d_g[u]:
        if p[w] == u and a[u] > a[w]:
            a[u] = a[w]
    return c

def p_o_a_driver(d_g, u=0):
    """
    Funcion que inicializa las tablas orden y ascenso de un grafo y las calcula

    	u: nodo en el que comenzamos
    	d_g (diccionario) lista de adyacencia del grafo a estudiar
    Returns:
	p, o, a: padres, orden y ascensos calculados
    """
    o = {}
    a = {}
    s = {}
    p = {}
    p[u] = None
    c = 0
    for node in d_g.keys():
        o[node] = np.inf
        a[node] = np.inf
        s[node] = False

    c = o_a_tables(u,d_g,p,s,o,a,c)
    return p,o,a

def hijos_bp(u,p):
    """
    Funcion que obtiene los hijos de un nodo mediante un diccionario de padres

    	u: nodo del que queremos obtener los hijos
    	p: diccionario de padres
    Returns:
	hijos: lista con los hijos de u
    """
    hijos = []
    for nodo in p.keys():
        if(p[nodo] == u):
            hijos.append(nodo)
    return hijos

def check_pda(p, o, a):
    """
    Funcion que obtiene los puntos de articulacion de un grafo

    	p: diccionario de padres
    	o: diccionario de orden de cada nodo
    	a: diccionario de ascenso de cada nodo
    Returns:
	pas: lista con los nodos que son puntos de articulacion
    """
    pas = []
    for nodo in p.keys():
        hijos = hijos_bp(nodo,p)
        if(p[nodo] == None):
            if(len(hijos)>1):
                pas.append(nodo)
        else:
            for hijo in hijos:
                if((o[nodo] <= a[hijo]) and (nodo not in pas)):
                    pas.append(nodo)
    return pas

def init_cd(d_g):
    """
    Funcion que inicializa un conjunto disjunto vacio con los nodos de un grafo

    	d_g (diccionario) lista de adyacencia del grafo del que queremos obtener el conjunto disjunto inicial
    Returns:
	d_cd: conjunto disjunto inicializado a -1
    """
    d_cd = {}
    for u in d_g.keys():
        d_cd[u] = -1

    return d_cd

def union(rep_1, rep_2, d_cd):
    """
    Funcion que une dos nodos en el conjunto disjunto

    	rep_1: representante 1
    	rep_2: representante 2
    	d_cd: conjunto disjunto donde queremos unir los dos representantes
    Returns:
	d_cd: conjunto disjunto nuevo
    """
    if(d_cd[rep_2] < d_cd[rep_1]):
        d_cd[rep_1] = rep_2
        return rep_2
    elif(d_cd[rep_2] > d_cd[rep_1]):
        d_cd[rep_2] = rep_1
        return rep_1
    else:
        d_cd[rep_2] = rep_1
        d_cd[rep_1] -= 1
        return rep_1
    return rep_1

def find(ind, d_cd, fl_cc):
    """
    Funcion que busca el representante de un nodo en el conjunto disjunto

    	ind: nodo a buscar
    	d_cd: conjunto disjunto donde queremos buscar
    	fl_cc: True si queremos aplicar compresion de caminos, False si no
    Returns:
	rep: representante del nodo buscado
    """
    rep = ind
    while (d_cd[rep] >= 0):
        rep = d_cd[rep]
    if fl_cc:
        while (d_cd[ind] >= 0):
            z = d_cd[ind]
            d_cd[ind] = rep
            ind = z
    return rep

def insert_pq(d_g, q):
    """
    Funcion que inserta las ramas de un grafo en una cola de prioridad ordenada por pesos de las ramas
    de menor a mayor

    	d_g (diccionario) grafo del que obtener las ramas
    	q: cola de prioridad donde insertaremos las ramas
    """
    for u in d_g.keys():
        for v in d_g[u].keys():
            if u < v:
                for elem in d_g[u][v].keys():
                    q.put((d_g[u][v][elem], (u, v)))

def kruskal(d_g, fl_cc=True):
    """
    Funcion que aplica el algoritmo de Kruskal a un grafo y devuelve el arbol abarcador minimo obtenido

    	d_g (diccionario) grafo donde aplicar el algoritmo
    	fl_cc: True si queremos aplicar compresion de caminos en la busqueda, False si no
    Returns:
	aam: arbol abarcador minimo del grafo obtenido, None si el grafo no era conexo
    """
    p = init_cd(d_g)
    cola = qe.PriorityQueue()
    colaAux = qe.PriorityQueue()
    insert_pq(d_g, cola)

    while not cola.empty():
        peso, uv = cola.get()
        u = uv[0]
        v = uv[1]

        colaAux.put((peso,uv))
    cola = colaAux
    aam = {u: {} for u in d_g}
    ramas = 0
    while not cola.empty():
        peso, uv = cola.get()
        u = uv[0]
        v = uv[1]

        x = find(u, p, fl_cc)
        y = find(v, p, fl_cc)
        if x != y:
            union(x, y, p)
            aam[u][v] = {0: peso}
            aam[v][u] = {0: peso}
            ramas +=1

    #para el caso de si no es conexo
    raiz = False
    for i in p.keys():
        if(raiz == False and p[i]<0):
            raiz = True
        elif(raiz == True and p[i]<0):
            return None

    # si es conexo, devolvemos el arbol
    return aam

def time_pda(n_graphs, n_nodes_ini, n_nodes_fin, step, prob):
    """
    Funcion que calcula los tiempos medios de ejecucion de la funcion p_o_a_driver con grafos generados aleatoriamente

    	n_graphs: numnero de grafos a crear de cada tipo
    	n_nodes_ini: numero de nodos inicial
    	n_nodes_fin: numero de nodos final
    	step: step de crecimiento del numero de nodos
    	prob: probabilidad de que existan ramas entre nodos de los grafos creados
    Returns:
	times: diccionario con claves el numero de nodos y valores el tiempo promedio de ejecucion de la funcion
    """
    grafos = {}
    times = {}
    for n_nodos in range(n_nodes_ini,n_nodes_fin+1,step):
        grafos[n_nodos] = {}
        for grafo in range(0,n_graphs):
            grafos[n_nodos][grafo] = rand_weighted_undirected_multigraph(n_nodos,prob=prob,max_weight=5,fl_diag=False)

    for n_nodos in grafos.keys():

        ini = time.time()
        for grafo in grafos[n_nodos].keys():
            p_o_a_driver(grafos[n_nodos][grafo])
        fin = time.time()
        times[n_nodos] = (fin - ini)/n_graphs
    return times

def time_kruskal(n_graphs, n_nodes_ini, n_nodes_fin, step, prob, fl_cc):
    """
    Funcion que calcula los tiempos medios de ejecucion de la funcion kruskal con grafos generados aleatoriamente

    	n_graphs: numnero de grafos a crear de cada tipo
    	n_nodes_ini: numero de nodos inicial
    	n_nodes_fin: numero de nodos final
    	step: step de crecimiento del numero de nodos
    	prob: probabilidad de que existan ramas entre nodos de los grafos creados
    	fl_cc: indica si queremos aplicar compresion de caminos o no
    Returns:
	times: diccionario con claves el numero de nodos y valores el tiempo promedio de ejecucion de la funcion
    """
    grafos = {}
    times = {}
    for n_nodos in range(n_nodes_ini,n_nodes_fin+1,step):
        grafos[n_nodos] = {}
        for grafo in range(0,n_graphs):
            grafos[n_nodos][grafo] = rand_weighted_undirected_multigraph(n_nodos,prob=prob,max_weight=1,fl_diag=False)

    for n_nodos in grafos.keys():
        time_aux = 0
        n_grafos = 0
        for grafo in grafos[n_nodos].keys():
            ini = time.time()
            ret = kruskal(grafos[n_nodos][grafo],fl_cc)
            fin = time.time()
            if(ret != None):
                time_aux+=fin-ini
                n_grafos+=1
        times[n_nodos] = time_aux/n_grafos
    return times

def kruskal_2(d_g, fl_cc=True):
    """Funcion que aplica el algoritmo de Kruskal a un grafo y devuelve el arbol abarcador minimo obtenido y el tiempo que tarda en crearlo (solo el tiempo de creacion del mismo)

    Args:
    	d_g (diccionario) grafo donde aplicar el algoritmo
    	fl_cc: True si queremos aplicar compresion de caminos en la busqueda, False si no

    Returns:
    	aam: arbol abarcador minimo del grafo obtenido, None si el grafo no era conexo
    	time: tiempo que ha tardado en crear el AAM, 0 si el grafo no es conexo
    """
    p = init_cd(d_g)
    cola = qe.PriorityQueue()
    colaAux = qe.PriorityQueue()
    insert_pq(d_g, cola)

    while not cola.empty():
        peso, uv = cola.get()
        u = uv[0]
        v = uv[1]

        colaAux.put((peso,uv))
    cola = colaAux
    aam = {u: {} for u in d_g}
    ramas = 0
    ini = time.time()
    while not cola.empty():
        peso, uv = cola.get()
        u = uv[0]
        v = uv[1]

        x = find(u, p, fl_cc)
        y = find(v, p, fl_cc)
        if x != y:
            union(x, y, p)
            aam[u][v] = {0: peso}
            aam[v][u] = {0: peso}
            ramas +=1
    fin = time.time()
    #para el caso de si no es conexo
    raiz = False
    for i in p.keys():
        if(raiz == False and p[i]<0):
            raiz = True
        elif(raiz == True and p[i]<0):
            return None,0.0

    # si es conexo, devolvemos el arbol
    return aam,(fin-ini)


def time_kruskal_2(n_graphs, n_nodes_ini, n_nodes_fin, step, prob, fl_cc):
    """Funcion que calcula los tiempos medios de ejecucion de la funcion kruskal_2 con grafos generados aleatoriamente

    Args:
        n_graphs: numero de grafos a crear de cada tipo
    	n_nodes_ini: numero de nodos inicial
    	n_nodes_fin: numero de nodos final
    	step: step de crecimiento del numero de nodos
    	prob: probabilidad de que existan ramas entre nodos de los grafos creados
    	fl_cc: indica si queremos aplicar compresion de caminos o no

    Returns:
	   times: diccionario con claves el numero de nodos y valores el tiempo promedio de ejecucion de la funcion
    """
    grafos = {}
    times = {}
    for n_nodos in range(n_nodes_ini,n_nodes_fin+1,step):
        grafos[n_nodos] = {}
        for grafo in range(0,n_graphs):
            grafos[n_nodos][grafo] = rand_weighted_undirected_multigraph(n_nodos,prob=prob,max_weight=1,fl_diag=False)

    for n_nodos in grafos.keys():
        time_aux = 0
        n_grafos = 0
        for grafo in grafos[n_nodos].keys():

            ret,time_aux = kruskal_2(grafos[n_nodos][grafo],fl_cc)

            if(ret != None):
                time_aux+=time_aux
                n_grafos+=1
        times[n_nodos] = time_aux/n_grafos
    return times
