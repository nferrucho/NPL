# para modificar los hiperparametros utilizamos modelos de busqueda
#idenficamos cual es el apropiado
from sklearn.model_selection import GridSearchCV

#definicion de cuadricula de hiperparametros
parametros = {
    'C' :[0,1,10,100],
    'kernel': ['linear', 'rbf'],
    'gamma' : ['scale','auto']
}

#creacion del objetivo para GridSearchCV
grid_search = GridSearchCV(estimator=svm_model, param_grid= parametros, cv=5)

#realizando el ajuste del modelo
grid_search.fit(X_entrena, y_entrena)

#obteniendo los hiperparametros optimos
mejor_parametro_grid = grid_search.best_params_
print(f'mejores_parametros = {mejor_parametro_grid}')
print()

A partir de programa anterior, puede indicarme que mas parametros puedo agregar al diccionario parametros