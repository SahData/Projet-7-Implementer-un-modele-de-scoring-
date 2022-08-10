# Projet-7-Implementer-un-modele-de-scoring-FastApi-Streamlit-Docker
Le projet consiste à développer pour la société « Prêt à Dépenser », une société́ de crédit de consommation, un modèle de scoring de la probabilité de défaut de paiement d’un client avec peu ou pas d’historique de prêt. 

Le projet a été traité à l’aide d’un modèle Baseline (Logistic Régression), et 3 algorithmes (XGBoost, LightGBM, et Random Forest) 

Un chargé de clientèle doit pouvoir utiliser le modèle via l’application mise à disposition, en face à face avec son client, dans le but de lui expliquer le plus simplement possible la décision envisagée dans l’étude de son dossier.

L’idée est donc d’expliciter au mieux le score renvoyé par le modèle. Pour réaliser ce module, la première perspective envisagée était d’utiliser l’importance des features issues des différents modèles utilisés. Puis un LIME qui est une librairie s’appliquant à n’importe quel modèle de machine Learning et permettant de comprendre comment évolue la prédiction d’un modèle et perturbant les variables en entrée du modèle. Ensuite un SHAP, pour une meilleure visualisation. 

