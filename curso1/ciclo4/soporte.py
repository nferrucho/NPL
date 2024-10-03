Requiero hacer una funcion en python llamada claustering, que deba entrenar un modelo K-means y evaluar el coeficiente de silueta para un numero especifico de cluster K.

Parametros:
features: arreglo de numpy con las características de los textos.
n_clusters: número de clusters a usar.
seed: semilla de números aleatorios.

Retorno_
model: modelo de K-Means entrenado.
score: valor del coeficiente de silueta.

Condiciones:
controlar el número de clusters con el parámetro n_clusters.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustering(features, n_clusters, seed):
  """
  Entrena un modelo K-Means y calcula el coeficiente de silueta.

  Args:
    features: Arreglo de NumPy con las características de los textos.
    n_clusters: Número de clusters a usar.
    seed: Semilla para la generación de números aleatorios.

  Returns:
    model: Modelo K-Means entrenado.
    score: Valor del coeficiente de silueta.
  """

  # Inicializar el modelo K-Means con la semilla especificada
  kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

  # Entrenar el modelo
  model = kmeans.fit(features)

  # Obtener las etiquetas de los clusters
  labels = model.labels_

  # Calcular el coeficiente de silueta
  score = silhouette_score(features, labels)

  return model, score
============================================
Realizar el entrenamiento de un vectorizador TF-IDF cpn sublinear scaling que permite extraer unicamento los terminos que aparecen
por lo menos en el 0.5% de los documentos del corpus.

Parámetros
corpus: pd.Series con los textos preprocesados del conjunto de datos.

Retorna:
features: arreglo de numpy con la representación de tipo TF-IDF.
vect: TfidfVectorizer entrenado con las especificaciones dadas.

condiciones
Recuerde que sublinear scaling se puede controlar con el parámetro sublinear_tf del vectorizador.
Recuerde convertir la representación a un arreglo de numpy.
Puede usar el parámetro min_df para filtrar términos por frecuencia de documento.

import pandas as pdf
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorizer(corpus):
  # Crear un vectorizador TF-IDF con las configuraciones especificadas
  vect = TfidfVectorizer(sublinear_tf=True, min_df=0.005)  # 0.5% de los documentos

  # Ajustar el vectorizador al corpus y transformar los datos
  features = vect.fit_transform(corpus)

  # Convertir la representación a un arreglo de NumPy denso
  features = features.toarray()

  return features, vect
=======================================
Requiero encontrar el documento más similar a un cluster en específico. El proceso debe seguir los siguientes pasos:

Calcular la similitud coseno entre las características de cada documento y el centroide de un cluster dado.
Encontrar el id del documento con mayor similitud coseno.
Extraer el documento del corpus.

Para esto debo implementar la función cluster_document, la cual toma como entrada el corpus, las características, un modelo entrenado y el id de un cluster. Esta función debe retornar el texto del documento más relevante.

Parámetros
corpus: pd.Series con el texto preprocesado.
features: arreglo de numpy con las características de los textos.
model: modelo de K-Means entrenado.
cluster_id: identificador del cluster a analizar.

Retorna:
relevant_doc: documento más relevante para el cluster en cuestión.

Condiciones
Puede acceder a los centroides del modelo K-Means con el atributo cluster_centers_ del modelo entrenado.
Puede usar la función np.argmax para encontrar el documento más similar.

def cluster_document(corpus, features, model, cluster_id):
  # Obtener el centroide del cluster específico
  centroid = model.cluster_centers_[cluster_id]

  # Calcular la similitud coseno entre cada documento y el centroide
  similarities = cosine_similarity(features, centroid.reshape(1, -1))

  # Encontrar el índice del documento con mayor similitud
  most_similar_doc_index = np.argmax(similarities)

  # Obtener el documento más relevante del corpus
  relevant_doc = corpus.iloc[most_similar_doc_index]

  return relevant_doc