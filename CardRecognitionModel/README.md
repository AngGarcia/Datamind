# Card-classifier

* Explicamos el proceso de los datos aunque no se pudieron subir a GitHub por temas de espacio.
* Creamos la carpeta 'Data' que contiene 'Raw por carpetas' con fotos separadas por carpetas y 'Labeled' con todas las fotos etiquetadas (tanto individuales como grupales).
* Creamos 'Create_augmented.py' que al ejecutar, aumenta la cantidad de fotos en 'Labeled' con las transformaciones que le indicamos. Crea también su bounding box y las guarda en 'Labeled_augmented'. Para comprobar que es correcto, puedes ejecutar 'Show_box.py' para ver los bounding boxes resultantes.
* Ahora tenemos que hacer dos cosas, que contemplamos en el fichero 'Preprocesado.py':
    * Dividir el dataset 'Labeled_augmented' en train y test.s
    * Los nombres de las clases están de esta forma (izquierda nombre dado por YOLO y derecha número de la carta):
    15 -> 01
    16 -> 02
    17 -> 03
    18 -> 04
    19 -> 05
    20 -> 06
    21 -> 07
    22 -> 08
    23 -> 09
    24 -> 10
    25 -> J
    26 -> Q
    27 -> K
    Tenemos que mapearlo para que la primera carta sea 0 y la última 12
* Ahora ya podemos usar 'Classes.yaml' que apunta a los datos de entrenamiento y test y dice qué carta es cada clase.
* Finalmente ejecutamos el archivo train.py para entrenar los datos. Se necesita GPU para poder crear el modelo en tiempo razonable.
* Una vez acabado de entrenar obtendremos la carpeta runs que contiene las métricas del modelo y dos modelos dentro de la carpeta 'weights', el modelo best.pt y last.pt. Utilizaremos el mejor para predecir.
* Para detectar las cartas de una foto, ejecutamos el archivo 'predict.py' con la foto que queremos inspeccionar.

