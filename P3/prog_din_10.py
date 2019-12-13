import random
import numpy as np
"""
	Practica 3
	Adrián Navas Ajenjo
	Gloria del Valle Cano
"""
def gen_super_crec(n_terms):
	'''Genera una sucesion supercreciente aleatoria
	Devuelve la sucesion generada
	
	Args:
		n_terms (int): numero de terminos que tendra la sucesion
		
	Returns:
		ssc (lista): lista que contiene los numeros de la sucesion supercreciente
	'''
	ssc = []
	last = random.randint(1,10)
	ssc.append(last)
	for i in range(1,n_terms):
		last = last+random.randint(last,last+10)
		ssc.append(last)
		
	return ssc

def multiplier(mod, mult_ini):
	""" Funcion que calcula el multiplicador cuyo valor va entre 1 y mod/2.
	Devuelve el multiplicador.

	Args:
		mod (int): modulo para el calculo del multiplicador
		mult_ini: valor de inicializacion (random)
	
	Returns:
		i(int): valor del multiplicador actualizado
	"""
	i = mult_ini+1
	while(True):
		if(mcd(i,mod) == 1):
			return i
		i+=1

def inverse(p, mod):
	'''Funcion que calcula el inverso multiplicativo de p en Zmod
	Esta funcion utiliza el algoritmo de Euclides extendido con el cual calculamos:
							1 = u*mod+v*p
	donde v es el inverso multiplicativo de p en Zmod.
	Devuelve el inverso multiplicativo en caso de que exista, y -1 si no existe
	
	Args:
		p (int): numero del que obtener el inverso multiplicativo
		mod (int): modulo en el que calcular el inverso multiplicativo
	
	Returns:
		vi (int): inverso multiplicativo de p en Zmod, -1 si no existe    
	'''
	# si el mcd no es uno, no tiene inv mult en Zmod
	if(mcd(p,mod) != 1):
		return -1
	
	r0 = mod # r(i-1)
	ri = p # r(i)
	u0 = 1 # u(i-1)
	v0 = 0 # v(i-1)
	ui = 0 # u(i)
	vi = 1 # v(i)
	
	# cuando ri=1 ya tenemos en vi el inv multiplicativo
	while(ri != 1):
		#calculamos el r(i+1)
		r2 = r0 % ri
		# y el qi
		qi = r0 // ri
		#calculamos los multiplicadores u y v para esta ronda
		u2 = u0-(qi*ui)
		v2 = v0-(qi*vi)
		
		#actualizamos los valores
		r0 = ri
		ri = r2
		u0 = ui
		ui = u2
		v0 = vi
		vi = v2
	
	return int(vi%mod)

def mod_mult_inv(l_sc):
	'''Funcion que calcula un multiplicador, su inverso y el modulo de la aritmetica en funcion
	a una clave privada
	Devuelve los tres valores mencionados.
	
	Args:
		l_sc (lista): lista que contiene la clave privada
	
	Returns:
		mul (int): multiplicador cuyo valor va entre 1 y mod/2
		inv (int): inverso multiplicativo de mul en Zmod
		mod (int): aritmetica a utilizar
		
	'''
	mod = sum(l_sc)+1
	mul = multiplier(mod,random.randint(1,int(mod/2)))
	inv = inverse(mul,mod)
	return mul,inv,mod
	
def gen_sucesion_publica(l_sc, p, mod):
	'''Funcion que calcula una clave publica en funcion a una privada, un multiplicador y un modulo
	Devuelve la clave privada
	
	Args:
		l_sc (lista): lista que contiene la clave privada
		p (int): multiplicador
		mod (int): aritmetica a utilizar 
	
	Returns:
		l_pub (lista): clave publica de la clave privada recibida calculada
		
	'''
	l_pub = []
	for elemento in l_sc:
		l_pub.append((p*elemento) % mod)
	return l_pub

def l_publica_2_l_super_crec(l_pub, q, mod):
	'''Funcion que calcula una clave privada en funcion a una publica, un inverso y un modulo
	Devuelve la clave privada
	
	Args:
		l_pub (lista): lista que contiene la clave publica
		q (int): inverso
		mod (int): aritmetica a utilizar 
	
	Returns:
		l_sc (lista): clave privada calculada (lista supercreciente)
		
	'''
	l_sc = []
	for elemento in l_pub:
		l_sc.append(((q*elemento) % mod))
	return l_sc

def mcd(a,b):
	'''Calcula el maximo comun divisor entre a y b utilizando el algoritmo de euclides
	Devuelve el maximo comun divisor entre ambos numeros
	
	Args:
		a (int): primer numero
		b (int): segundo numero
		
	Returns:
		mcd (int): maximo comun divisor entre a y b
	'''
	if(a<b):
		a_aux = a
		a = b
		b = a_aux
		
	if((a % b) == 0):
		return b
	return mcd(b,a%b)

def gen_random_bit_list(n_bits):
	'''Funcion que genera una lista aleatoria de bits
	
	Args:
		n_bits (int): numero de bits que contendra la lista generada 
	
	Returns:
		l_bits (lista): lista de bits generada
		
	'''
	l_bits = []
	for i in range(0,n_bits):
		l_bits.append(random.choice([0,1]))
	return l_bits

def mh_encrypt(l_bits, l_pub, mod):
	'''Funcion que cifra una cadena de bits utilizando una clave publica y el modulo
	
	Args:
		 l_bits (lista): lista de bits a cifrar
		 l_pub (lista): lista que contiene la clave publica
		 mod (int): aritmetica a utilizar 
	
	Returns:
		encrypted_blocks (lista): lista de enteros donde cada entero contiene un bloque cifrado
	'''
	encrypted_blocks = []
	block_size = len(l_pub)
	#Anadimos 0s al final hasta que sea de la longitud de l_pub
	while((len(l_bits) % block_size) != 0):
		l_bits.append(0)
	for i in range(0,len(l_bits),block_size):
		res = 0
		k=0
		for ki in l_pub:
			res+=ki * l_bits[i+k]
			k+=1
		encrypted_blocks.append(res)
		
	return encrypted_blocks

def mh_block_decrypt(c, l_sc, inv, mod):
	'''Funcion que descifra un entero utilizando una clave privada, un inverso y el modulo
	
	Args:
		 c (int): entero que contiene un bloque cifrado
		 l_sc (lista): lista supercreciente con la clave privada
		 inv (int): inverso
		 mod (int): aritmetica a utilizar 
	
	Returns:
		decrypted_block (lista): lista de bits con el bloque descifrado
	'''
	
	# creamos una lista vacia con el tamano de la clave privada
	decrypted_block = [None] * len(l_sc)
	# guardamos en la variable resto el valor del bloque codificado * inverso modulo mod
	resto = (c*inv) % mod
	
	# creamos el diccionario numMapper que mapea los valores de la clave privada con su indice
	# para despues ir anadiendo los bits al bloque decodificado de forma ordenada
	numMapper = {}
	i=0
	for num in l_sc:
		numMapper[num] = i
		i+=1
 
	# aplicamos el problema de la suma en la clave publica ordenada de mayor a menor
	for ki in sorted(l_sc,reverse=True):
		if(resto>=ki):
			# si el numero entra, le anadimos un uno al bloque
			decrypted_block[numMapper[ki]] = 1
			# y actualizamos el resto con el numero restante
			resto -= ki
		else:
			# si no entra el numero en el numero estudiado, anadimos un cero y continuamos
			decrypted_block[numMapper[ki]] = 0
	
	return decrypted_block


def mh_decrypt(l_cifra, l_sc, inv, mod):
	'''Funcion que descifra un texto cifrado utilizando una clave privada, un inverso y el modulo
	
	Args:
		 l_cifra: lista que contiene los bloques cifrados (texto completo)
		 l_sc (lista): lista supercreciente con la clave privada
		 inv (int): inverso
		 mod (int): aritmetica a utilizar 
	
	Returns:
		blocks (lista): lista de bits con el texto descifrado
	'''
	blocks = []
	for c in l_cifra:
		blocks += mh_block_decrypt(c, l_sc, inv, mod)
		
	return blocks

def min_coin_dict(c, l_coins):
	"""Calcula el diccionario para el numero minimo de monedas de l_coin necesarias para dar cambio de una cantidad c.
	Devuelve la matriz con las subsoluciones hasta el cambio final
	
	Args:
		c (int): cantidad a dar cambio
		l_coins (lista): lista de enteros positivos que contiene las monedas
	
	Returns:
		m: matriz necesaria para la obtención del mínimo número de monedas para dar cambio de una cantidad c
	"""
	assert(c>0 and (1 in l_coins) and all(i>0 for i in l_coins))
	m = {i: {j:0 for j in range(c+1)} for i in range(len(l_coins))}
	
	for i in range(1, c+1): 
		m[0][i] = i
		
	for i in range(1, len(l_coins)):
		m[i][0] = 0 
		
	for i in range(1, len(l_coins)):
		for j in range(1, c+1):
			res = j - l_coins[i]
			if (res >= 0): 
				m[i][j] = min(m[i-1][j], 1 + m[i][j-l_coins[i]])
			else:
				m[i][j] = m[i-1][j]
	return m

def min_coin_number(c, l_coins):
	"""Calcula el numero minimo de monedas de l_coin necesarias para dar cambio de una cantidad c.
	Devuelve el valor calculado
	
	Args:
		c (int): cantidad a dar cambio
		l_coins (lista): lista de enteros positivos que contiene las monedas
	
	Returns:
		m[i][j] (int): mínimo número de monedas para dar cambio de una cantidad c
	"""
	m = min_coin_dict(c, l_coins)
	return m[len(l_coins)-1][c]

def optimal_change(c, l_coins):
	"""Calcula el numero optimo de monedas de l_coin necesarias para dar cambio de una cantidad c
	Devuelve un diccionario con las monedas de cada tipo en l_coin necesarias para dar cambio de una cantidad c
	
	Args:
		c (int): cantidad a dar cambio
		l_coins (lista): lista de enteros positivos que contiene las monedas
	
	Returns:
		d: diccionario con el número de monedas necesario para dar cambio de una cantidad c
	"""
	l = len(l_coins) -1
	l_coins = sorted(l_coins)
	m_d = min_coin_dict(c, l_coins)
	d = {i:0 for i in l_coins}
	while l >= 0 and c > 0:
		if l > 0 and m_d[l][c] == m_d[l-1][c]:
			l -= 1
		else:
			c -= l_coins[l]
			d[l_coins[l]] += 1
	return d

def max_matrix_common_subsequence(str_1, str_2):
	"""Calcula la matriz de subsecuencia común más larga entre dos cadenas
	Devuelve la matriz de longitudes de subsecuencias máximas comunes parciales
	
	Args:
		str_1: cadena 1
		str_2: cadena 2
	
	Returns:
		m: matriz de longitudes de subsecuencias máximas comunes parciales
	"""
	assert(len(str_1)>1 and len(str_2)>1)
	l1, l2 = len(str_1)+1, len(str_2)+1
	m = [[0 for j in range(l1)] for i in range(l2)]

	for i in range(1, l2):
		for j in range(1, l1):
			if i == 0 or j == 0:
				m[i][j] = 0
			elif str_1[j-1] == str_2[i-1]:
				m[i][j] = m[i-1][j-1] + 1
			else:
				m[i][j] = max(m[i-1][j], m[i][j-1])            
	return m


def max_length_common_subsequence(str_1, str_2):
	"""Calcula la matriz de subsecuencia común más larga entre dos cadenas
	Devuelve la matriz de longitudes de subsecuencias máximas comunes parciales
	
	Args:
		str_1: cadena 1
		str_2: cadena 2
	
	Returns:
		m: matriz de longitudes de subsecuencias máximas comunes parciales
	"""
	i, j = len(str_2), len(str_1)
	m = max_matrix_common_subsequence(str_1, str_2)
	return m[i][j]

def find_max_common_subsequence(str_1, str_2):
	"""Encuentra la subsecuencia común de longitud máxima
	Devuelve la subsecuencia común de longitud máxima
	
	Args:
		str_1: cadena 1
		str_2: cadena 2
	
	Returns:
		cad: array que contiene la subsecuencia común máxima
	"""
	i, j = len(str_2), len(str_1)
	m = max_matrix_common_subsequence(str_1, str_2)
	ind = max_length_common_subsequence(str_1, str_2)
	cad = [""]*ind
	while (i > 0 and j > 0):
		if str_1[j-1] == str_2[i-1]:
			cad[ind-1] = str_1[j-1]
			i -= 1
			j -= 1
			ind = ind-1
		elif m[i-1][j] > m[i][j-1]:
			i -= 1
		else:
			j -= 1
	return cad
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
	
	minimo = np.inf
	root = np.inf
	# guardamos la suma de las probabilidades entre L y R
	sum_probs = np.sum(m.diagonal()[L:R+1])
	
	# ponemos la raiz en cada i para estudiar el arbol minimo
	for i in range(L,R+1):
		# si la i esta en L estudiamos solo TR
		if(i == L):
			if(minimo > m[i+1,R]):
				minimo = m[i+1,R]
				root = i
		# si la i esta en R estudiamos solo TL
		elif(i == R):
			if(minimo > m[L,i-1]):
				minimo = m[L,i-1]
				root = i
		# si no, estudiamos TL + TR con raiz en i
		else:
			if(minimo > m[L,i-1]+m[i+1,R]):
				minimo = m[L,i-1]+m[i+1,R]
				root = i
			
	return (minimo+sum_probs),root

def optimal_order(l_probs):
	"""Funcion que encuentra el arbol minimo entre L y R probando todas las raices posibles
	y usando la matriz m
	Devuelve el valor minimo de C(T LR)
	
	Args:
		l_probs (lista): lista con las probabilidades de cada nodo a insertar en el arbol
		
	Returns:
		m (np.matrix): matriz con los costes minimos de los subarboles
	"""
	assert(sum(l_probs) == 1)
	num_nodos = len(l_probs)
	m = np.zeros((num_nodos,num_nodos))
	m_roots = np.zeros((num_nodos,num_nodos),dtype='int32')
	m.fill(np.inf)
	
	# rellenamos la diagonal con las probabilidades de cada nodo
	for i in range(0,num_nodos):
		m[i,i] = l_probs[i]
	
	# estudiamos la parte superior de la diagonal de la matriz
	for i in range(1,num_nodos):
		for j in range(num_nodos):
			if(j+i < num_nodos):
				m[j,j+i], m_roots[j,j+i] = get_min_tree(m,j,j+i)         
	return m,m_roots

def list_opt_ordering_search_tree(m_roots, l, r):
	""" Funcion que encuentra la lista con el orden de insercion de las claves l, l+1, ..., r en el BST
	Devuelve la lista
	
	Args:
		m_roots(np.matrix): la matriz de raices devuelta por la funcion optimal_order
		l: indices l (izquierdo)
		r: indices r (derecho)
	
	Returns:
		Lista con el orden de insercion de las claves l, l+1, ..., r en el 
			correspondiente arbol binario de busqueda optimo.
	"""
	if(l > r):
		return []
	elif l == r:
		return [l]
	k = m_roots[l,r]
	return [k] + list_opt_ordering_search_tree(m_roots,l,k-1) + list_opt_ordering_search_tree(m_roots,k+1,r)

m, m_roots = optimal_order([0.1,0.2,0.3,0.4])
print(m, m_roots)