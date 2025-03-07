# vqa_project
Project for the Neural Networks module, part of the NLP masters at University of Paris Nanterre

## Téléchargement des données
Pour télécharger les données textuelles (les questions et les choix multiples):
```
cd data
bash download_dataset.sh
```
_Attention! Si vous n'utilisez pas macOS, vous devez ajuster le script, en suivant les instructions au debut du script._

Au cas où le téléchargement avec le script bash ne marche pas, utilisez simplement le lien qui se retrouve dans le script.

*Les données sont sous MIT License, comme indiqué [ici](https://github.com/yukezhu/visual7w-toolkit/tree/master?tab=MIT-1-ov-file)*

Pour télécharger les images associés aux questions:
```
cd data
bash download_images.sh
```
Attention! Il s'agit de 43000 images en format .jpg, donc le temps de téléchargement peut être assez long. 

## Data processing
Ensuite, avant de lancer l'entrainement et l'évaluation du modele lancez d'abbord le script `get_data.py` dont l'output sera une dataframe sauvegardé au format pickle à ce chemin: `data/final_clean_df.pkl`

Ce script permet d'obtenir les embeddings des images et du contenu textuel, ainsi que de transformer les données dans un format utilisable par un réseau de neurones.

## Entrainement et evaluation du modèle

Une fois que les données ont été processées et le fichier pickle est sauvegardé, vous pouvez lancer le script `train_model.py` en utilisant des arguments en ligne de commande:

### Usage: 
```
python train_model.py --model <model_name> --epochs <num_epochs> --save_model
```

- model_name: Model architecture to use (NetSimple, CoAttentionNetSimple, CoAttentionNetBNDropout)
- num_epochs: Number of epochs to train the model (default: 10)
- save_model: Save the trained model (optional)

Les caracteristiques de chaque architecture sont visible dans le script `datastructure.py`