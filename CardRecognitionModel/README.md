# Card-classifier

* Creamos la carpeta 'Data' que contiene 'Raw por carpetas' con fotos individuales y grupales por carpetas y 'Labeled' con TODAS las fotos labeled (individuales y grupales)
* Creamos 'Create_augmented.py' que al ejecutar, aumenta la cantidad de fotos en 'Labeled' con las transformaciones que le indicamos y crea automáticamente su bounding box y las guarda en 'Labeled_augmented'. Para comprobar que es correcto, puedes ejecutar 'Show_box.py' para ver los bounding boxes resultantes.
* Ahora tenemos que hacer dos cosas, que contemplamos en el fichero 'Preprocesado.py':
    * Dividir el dataset 'Labeled_augmented' en train y test
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
* Ahora ya podemos usar 'Classes.yaml' que apunta a los datos de entrenamiento y test y dice qué carta es cada clase. (01 es As).

