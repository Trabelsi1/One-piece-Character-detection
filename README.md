# One Piece Character Detection

Projet de vision par ordinateur pour la detection d'objets de personnages de One Piece (membres de l'equipage de Luffy) a partir d'images.

## Presentation du projet

L'objectif de ce projet est de construire un modele de detection capable d'identifier automatiquement plusieurs personnages dans une image (ou une frame video).

Ce depot contient actuellement:
- Le dataset annote (splits train/valid/test)
- Les metadonnees d'export Roboflow
- La base documentaire du projet

## Objectifs

- Entrainer un detecteur multi-classes de personnages
- Evaluer les performances (precision, rappel, mAP)
- Pouvoir reutiliser le modele pour de la prediction sur de nouvelles images
- Structurer un pipeline simple et reproductible pour l'equipe

## Personnages cibles (classes)

1. Luffy
2. Nami
3. NicoRobin
4. Usopp
5. Zoro
6. sanji

## Donnees

Informations principales du dataset:
- Total: 3409 images
- Format d'annotation: YOLO v8 (`.txt`)
- Taille des images: 512x512
- Source: Roboflow Universe

Repartition:
- Train: 3213 images
- Validation: 98 images
- Test: 98 images

Pretraitements appliques:
- Auto-orientation (EXIF retire)
- Redimensionnement a 512x512 (stretch)

Augmentation utilisee:
- Rotations a 90 degres (aucune / horaire / anti-horaire)

## Structure du depot

```
.
|-- README.md
|-- README.dataset.txt
|-- data.yaml
|-- README.roboflow.txt
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
`-- test/
    |-- images/
    `-- labels/
```

## Format des annotations

Chaque image possede un fichier `.txt` associe, avec une ligne par objet:

```
<class_id> <x_center> <y_center> <width> <height>
```

Les coordonnees sont normalisees dans l'intervalle `[0, 1]`.


## Pistes d'amelioration

- Ajouter plus d'images pour les classes sous-representees
- Enrichir les scenes (angles, eclairages, occlusions)
- Tester des modeles plus recents et comparer les performances


## Source et attribution

- Roboflow Universe: https://universe.roboflow.com/trabelsis-workspace-kurma/one-piece-eyhhn
- Licence du dataset: CC BY 4.0

