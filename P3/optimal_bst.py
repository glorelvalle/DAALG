def get_min_tree(m,L,R):
    """Funcion que encuentra el arbol minimo entre L y R probando todas las raices posibles
    y usando la matriz m
    Devuelve el valor minimo de C(T LR)
    
    Args:
        m (np.matrix): matriz que contiene los subarboles minimos
        L (int): nodo izquierdo
        R (int): nodo derecho
        
    Returns:
        (float): valor del coste minimo del arbol entre L y R
    """
    minimo = float("inf")
    root = float("inf")
    # guardamos la suma de las probabilidades entre L y R
    sum_probs = 0
    
    # ponemos la raiz en cada i para estudiar el arbol minimo
    for i in range(L,R+1):
        # si la i esta en L estudiamos solo TR
        if(i == L):
            if(minimo > m[i+1][R]):
                minimo = m[i+1][R]
                root = i
        # si la i esta en R estudiamos solo TL
        elif(i == R):
            if(minimo > m[L][i-1]):
                minimo = m[L][i-1]
                root = i
        # si no, estudiamos TL + TR con raiz en i
        else:
            if(minimo > m[L][i-1]+m[i+1][R]):
                minimo = m[L][i-1]+m[i+1][R]
                root = i
            
    return (minimo+sum_probs),root

def optimal_order(l_probs):
    """Funcion que encuentra el arbol minimo entre L y R probando todas las raices posibles
    y usando la matriz m
    Devuelve el valor minimo de C(T LR)
    
    Args:
        l_probs (lista): lista con las probabilidades de cada nodo a insertar en el arbol
        
    Returns:
        m (array): matriz con los costes minimos de los subarboles
    """
    assert(sum(l_probs) == 1)
    num_nodos = len(l_probs)
    m = [[0 for i in range(num_nodos)] for j in range(num_nodos)]
    m_roots = [[-1 for i in range(num_nodos)] for j in range(num_nodos)]
    
    # rellenamos la diagonal con las probabilidades de cada nodo
    for i in range(0,num_nodos):
        m[i][i] = l_probs[i]
    
    # estudiamos la parte superior de la diagonal de la matriz
    for i in range(1,num_nodos):
        for j in range(num_nodos):
            if(j+i < num_nodos):
                m[j][j+i], m_roots[j][j+i] = get_min_tree(m,j,j+i)         
    return m,m_roots