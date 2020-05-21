# Projet-IA (SNIPS)

Dans ce projet, l'objectif est la conception et l'implémentation d'un système purement neuronnal qui réalise la détection du mot d'éveil. L'implementation se fera en avec le frameowrk Pytorch.

## Les données

Les données ont été produites et sont distribuées par la société Snips. Plus d'information [ici](https://github.com/snipsco/keyword-spotting-research-datasets)

## PyTorch-Lightning

 Pour la réalisation de ce projet, nous avons utilisé le wrapper [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning). Lightning permet principalement d'organiser le code PyTorch et d'avoir accès à des fonctionnalités supplémentaire tel que le multi-GPU, la précision 16 bits nativement...

![cmp](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/docs/source/_images/lightning_module/pt_to_pl.png?raw=true)


## Paramétrisation acoustique

Afin d'extraire les informations pertinentes des signaux audio, nous avons choisi de travailler avec les **MFCC** (Mel-frequency cepstral coefficients). Se choix a été motivé par les conseils de Mr Esteve ainsi que l'experience que nous avons avec cette approche.

### MFCC

PyTorch met à disposition une méthode pour le calcule des **MFCC** [`torchaudio.transforms.MFCC`](https://pytorch.org/audio/_modules/torchaudio/transforms.html), cependant les paramètres ne sont pas entièrement configurable. Nous avons donc utilisé [`torchaudio.compliance.kaldi`](https://pytorch.org/audio/compliance.kaldi.html) qui met à disposition des fonctions identique à celle de **Kaldi**.

> **Config:**
> ```ini
> [MFCC]
>n_fft = 400
>frame_length = 20
>frame_shift = 10
> # [-1, 0]
>channel = -1
># [0.0, 1.0]
>dither = 0.0
>#['hanning', 'povey']
>window_type = povey
>sample_frequency = 16000
>num_ceps = 13
>num_mel_bins = 23
>snip_edges = False
> ```

### Pré-traitement

Afin de limiter l'impact de certains phénomènes apparaissant dans le signal de parole, nous avons appliqué une CMVN (Cepstral Mean Variance Normalization) par uttérance. Nous aurions aussi pu effectuer une normalisation par speaker et mesurer l'impact sur les perfomances du système.


## Modèle 

```
  | Name    | Type        | Params
------------------------------------
0 | rnn     | GRU         | 248 K
1 | fc1     | Linear      | 409 K
2 | bn1     | BatchNorm1d | 128
3 | dropout | Dropout     | 0
4 | fc4     | Linear      | 65

```

Concernant le choix de l'architecture, nous avons opté pour une approche récurente à base de **GRU**. Dans les articles que nous avons consulté, nous avons souvent retrouvé des modèles récurents ou convolutif, leurs performances sont similaires. Il existe aussi des approches combinant les deux approches qui obtiennent de meilleurs résultats.

Notre choix pour pour les **GRU** vient aussi du fait que nous voulions approfondir nos connaissance avec ce type d'architecture.

## Expérience



## Résultats