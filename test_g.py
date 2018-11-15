''' Import dataset '''
import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree 


iris = load_iris()
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]

#	for i in range(len(iris.target)):
#	print "Example %d: label %s, features %s" %(i, iris.target[i], iris.data[i])

''' training data : rimuovo dal dataset gli elementi 0, 50 e 100 perchè li userò
	successivamente come test set'''

test_idx = [0,50,100]
train_target = np.delete (iris.target, test_idx)
train_data	= np.delete(iris.data, test_idx, axis=0)



'''	testing data: il test_data sarà tutto iris meno le 3 tuple 0,50,100 usate
	per effettuare il test '''
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

''' COSTRUISCO IL MODELLO '''
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

''' stampo il target di cui gia ero a conoscenza per fare la verifica... '''
print test_target

''' EFFETTUO LA PREVISIONE '''
print clf.predict(test_data)

''' ESPORTO IN PDF IL RISULTATO DELL'ALBERO DECISIONALE'''
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
	out_file=dot_data,
	feature_names=iris.feature_names,
	class_names=iris.target_names,
	filled=True, rounded=True,
	impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
