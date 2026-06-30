#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2025/2026
#############################################################################

### Домашно задание 1
###
### За да работи програмата трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
### предоставен за некомерсиално ползване от Института за български език - БАН
###
### Корпусът може да бъде свален от:
### Отидете на http://dcl.bas.bg/BulNC-registration/#feeds/page/2
### И Изберете:
###
### Корпус с новини
### Корпус от публицистични текстове за Югоизточна Европа.
### 27.07.2012 Български
###	35337  7.9M
###
### Архивът трябва да се разархивира в директорията, в която е програмата.
###
### Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
### conda activate tii
###
### Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции

import langmodel
import math
import numpy as np

def editDistance(s1 : str, s2 : str) -> np.ndarray:
	#### функцията намира модифицираното разстояние на Левенщайн между два низа, описано в условието на заданието
	#### вход: низовете s1 и s2
	#### изход: матрицата M с разстоянията между префиксите на s1 и s2 (виж по-долу)

	M = np.zeros((len(s1)+1,len(s2)+1))
	#### M[i,j] следва да съдържа разстоянието между префиксите s1[:i] и s2[:j]
	#### M[len(s1),len(s2)] следва да съдържа разстоянието между низовете s1 и s2
	#### За справка разгледайте алгоритъма editDistance от слайдовете на Лекция 1
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 10-30 редаs

	longerWordLenght = len(s1) if len(s1) >= len(s2) else len(s2) 

	for i in range(len(s1) + 1):
		M[i, 0] = i
	
	for j in range(len(s2) + 1):
		M[0, j] = j

	for i in range(1, len(s1) + 1):
		for j in range(1, len(s2) + 1):

			x = 0 if s1[-i] == s2[-j] else 1
			a = [1 + M[i - 2, j - 1] if i - 2 >= 0 else longerWordLenght + 1, #merge
				 1 + M[i - 1, j - 2] if j - 2 >= 0 else longerWordLenght + 1, #split
				 1 + M[i, j - 1],  											  #insert
				 1 + M[i - 1, j], 											  #delete
				 x + M[i - 1, j - 1]]										  #substitute
			M[i, j] = np.min(a)
			
	#### Край на Вашия код
	#############################################################################

	return M

def editWeight(s1 : str, s2 : str, Weight : dict[tuple[str,str],float]) -> float:
	#### функцията editWeight намира теглото между два низа
	#### вход: низовете s1 и s2, както и речник Weight, съдържащ теглото на всяка от елементарните редакции 
	#### изход: минималната сума от теглата на елементарните редакции, необходими да се получи от единия низ другия
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда
	
	Weight[('', '')] = 0

	maxWeight = 0
	for (c, b) in Weight:
		if Weight[(c, b)] > maxWeight:
			maxWeight = Weight[(c, b)]
	
	longer = len(s1) if len(s1) >= len(s2) else len(s2)
	G = np.zeros((len(s1) + 1, len(s2) + 1))

	G[0, 0] = 0

	for i in range(1, len(s1) + 1):
		G[i, 0] = G[i - 1, 0] + Weight[(s1[-i], "")]

	for j in range(1, len(s2) + 1):
		G[0, j] = G[0, j - 1] + Weight[("", s2[-j])]

	for i in range(1, len(s1) + 1):
		for j in range(1, len(s2) + 1):

			a = [Weight[(s1[-i], s2[-j])] + G[i - 1, j - 1],  #subsitute
				 Weight[(s1[-i], "")] + G[i - 1, j], #delete
				 Weight[("", s2[-j])] + G[i, j - 1], #insert
				 Weight[(s1[-i : -i + 2], s2[-j])] + G[i - 2, j - 1] if i > 1 and s1[-i + 1] != s2[-j] and s1[-i] != s2[-j] else maxWeight * longer + 1, #merge
				 Weight[(s1[-i], s2[-j : -j + 2])] + G[i - 1, j - 2] if j > 1 and s1[-i] != s2[-j + 1] and s1[-i] != s2[-j] else maxWeight * longer + 1] #split
			G[i, j] = np.min(a)

	return G[len(s1), len(s2)]

	#### Край на Вашия код
	#############################################################################


def bestAlignment(s1 : str, s2 : str) -> list[tuple[str,str]]:
	#### функцията намира подравняване с минимално тегло между два низа 
	#### вход: 
	####	 низовете s1 и s2
	#### изход: 
	####	 списък от елементарни редакции, подравняващи s1 и s2 с минимално тегло


	M = editDistance(s1, s2)
	alignment = []
	
	#############################################################################
	#### УПЪТВАНЕ:
	#### За да намерите подравняване с минимално тегло следва да намерите път в матрицата M,
	#### започващ от последния елемент на матрицата -- M[len(s1),len(s2)] до елемента M[0,0]. 
	#### Всеки преход следва да съответства на елементарна редакция, която ни дава минимално
	#### тегло, съответстващо на избора за получаването на M[i,j] във функцията editDistance.
	#### Събирайки съответните елементарни редакции по пъта от M[len(s1),len(s2)] до M[0,0] 
	#### в обратен ред ще получим подравняване с минимално тегло между двата низа.
	#### Всяка елементарна редакция следва да се представи като двойка низове.
	#### ПРЕМЕР:
	#### bestAlignment('редакция','рдашиа') = [('р','р'),('е',''),('д' 'д'),('а','а'),('кц','ш'),('и','и'),('я','а')]
	#### ВНИМАНИЕ:
	#### За някой двойки от думи може да съществува повече от едно подравняване с минимално тегло.
	#### Достатъчно е да изведете едно от подравняванията с минимално тегло.
	#############################################################################	
	
	#############################################################################	
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда

	i, j = len(s1), len(s2)

	while i > 0 or j > 0:
		candidate = []
		if j > 0 and i > 0:
			candidate.append((i - 1, j - 1, M[i - 1, j - 1], s1[-i], s2[-j]))
		if i > 0:
			candidate.append((i - 1, j, M[i - 1, j], s1[-i], ""))
		if j > 0:
			candidate.append((i, j - 1, M[i, j - 1], "", s2[-j]))

		if j > 1 and i > 0 and s1[-i] != s2[-j + 1] and s1[-i] != s2[-j]:
			if j > 2:
				candidate.append((i - 1, j - 2, M[i - 1, j - 2], s1[-i], s2[-j:(-j + 2)]))
			else:
				candidate.append((i - 1, j - 2, M[i - 1, j - 2], s1[-i], s2[-j:]))

		if i > 1 and j > 0 and s1[-i + 1] != s2[-j] and s1[-i] != s2[-j]:
			if i > 2: 
				candidate.append((i - 2, j - 1, M[i - 2, j - 1], s1[-i: (-i + 2)], s2[-j]))
			else:
				candidate.append((i - 2, j - 1, M[i - 2, j - 1], s1[-i:], s2[-j]))

		lowest = min(candidate, key = lambda x: x[2])
		alignment.append((lowest[3], lowest[4]))
		i = lowest[0]
		j = lowest[1]

	#### Край на Вашия код
	#############################################################################			
	return alignment

def trainWeights(corpus : list[tuple[str,str]]) -> dict[tuple[str,str],float]:
	#### Функцията editionWeights връща речник съдържащ теглото на всяка от елементарните редакции
	#### Функцията реализира статистика за честотата на елементарните редакции от корпус, състоящ се от двойки сгрешен низ и коригиран низ. Теглата са получени след оценка на вероятността за съответната грешка, използвайки принципа за максимално правдоподобие.
	#### Вход: Корпус от двойки сгрешен низ и коригиран низ
	#### изход: речник съдържащ теглото на всяка от елементарните редакции
	
	ids = subs = ins = dels = splits = merges = 0
	for q,r in corpus:
		alignment = bestAlignment(q,r)
		for op in alignment:
			if len(op[0]) == 1 and  len(op[1]) == 1 and op[0] == op[1]: ids += 1
			elif len(op[0]) == 1 and  len(op[1]) == 1: subs += 1
			elif len(op[0]) == 0 and  len(op[1]) == 1: ins += 1
			elif len(op[0]) == 1 and  len(op[1]) == 0: dels += 1
			elif len(op[0]) == 1 and  len(op[1]) == 2: splits += 1
			elif len(op[0]) == 2 and  len(op[1]) == 1: merges += 1
	N = ids + subs + ins + dels + splits + merges

	weight = {}
	for a in langmodel.alphabet:
		weight[(a,a)] = - math.log( ids / N )
		weight[(a,'')] = - math.log( dels / N )
		weight[('',a)] = - math.log( ins / N )
		for b in langmodel.alphabet:
			if a != b:
				weight[(a,b)] = - math.log( subs / N )
			for c in langmodel.alphabet:
				if a != c and b != c:
					weight[(a+b,c)] = - math.log( merges / N )
					weight[(c,a+b)] = - math.log( splits / N )

	return weight


def generateEdits(q : str) -> list[str]:
	### помощната функция, generate_edits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
	### Вход: заявка като низ q
	### Изход: Списък от низове с модифицирано разстояние на Левенщайн 1 от q
	###
	### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана в langmodel.alphabet
	###
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 10-20 реда

	a = []
	
	for i in range(len(q)):
		for c in langmodel.alphabet:
			if q[i] != c:
				a.append(q[:i] + c + q[i + 1:])       # substitute
				a.append(q[:i] + c + q[i:])           # insert
			if i > 0:
				a.append(q[:i - 1] + c + q[i + 1:])   # merge

	for i in range(len(q)):
		a.append(q[:i] + q[i + 1:])					  # delete
	
	for i in range(len(q)): 
		for c1 in langmodel.alphabet:
			for c2 in langmodel.alphabet:
				a.append(q[:i] + c1 + c2 + q[i + 1:]) #  split

	return a

	#### Край на Вашия код
	#############################################################################


def generateCandidates(query : str, dictionary : dict[str, int]) -> list[str]:
	### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, 
	# за да се получат кандидатите за корекция. 
	# Връщат се единствено кандидати, за които всички думи са в речника dictionary.
		
	### Вход:
	###	 Входен низ: query
	###	 Речник: dictionary

	### Изход:
	###	 Списък от низовете, които са кандидати за корекция
	
	def allWordsInDictionary(q : str) -> bool:
		### Помощна функция, която връща истина, ако всички думи в заявката са в речника
		return all(w in dictionary for w in q.split())


	L=[]
	if allWordsInDictionary(query):
		L.append(query)
	A = generateEdits(query)
	pb = langmodel.progressBar()
	pb.start(len(A))
	for query1 in A:
		if allWordsInDictionary(query1):
			L.append(query1)
		pb.tick()
		for query2 in generateEdits(query1):
			if allWordsInDictionary(query2):
				L.append(query2)
	pb.stop()
	return L


def correctSpelling(r : str, model : langmodel.MarkovModel, weights : dict[tuple[str,str],float], mu : float = 1.0, alpha : float = 0.9):
	### Комбинира вероятността от езиковия модел с вероятността за редактиране на кандидатите за корекция, 
	# генерирани от generate_candidates за намиране на най-вероятната желана (коригирана) заявка по дадената оригинална заявка query.
	###
	### Вход:
	###		заявка: r,
	###		езиков модел: model,
	###	 речник съдържащ теглото на всяка от елементарните редакции: weights
	###		тегло на езиковия модел: mu
	###		коефициент за интерполация на езиковият модел: alpha
	### Изход: най-вероятната заявка


	### УПЪТВАНЕ:
	###	Удачно е да работите с логаритъм от вероятностите. Логаритъм от вероятността
	#от езиковия модел може да получите като извикате метода model.sentenceLogProbability. 
	#Минус логаритъм от вероятността за редактиране може да получите като извикате функцията editWeight.
	#############################################################################
	#### Начало на Вашия код за основното тяло на функцията correct_spelling. На мястото на pass се очакват 3-10 реда

	dictionary = {k[0].replace(langmodel.startToken, "") for k in list(model.Tc.keys())[1:]}
	candidates = generateCandidates(r, model.kgrams[tuple()]) 
	best_score = float("-inf")
	best = None

	for c in candidates:
		lang = model.sentenceLogProbability(list(c.split()), alpha)
		log = - editWeight(r, c, weights)

		score = mu * lang + alpha * log

		if score > best_score:
			best_score = score
			best = c

	return best


	#### Край на Вашия код
	#############################################################################


