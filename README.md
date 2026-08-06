# Trading Assistant Bruno — prototype gratuit

Ce projet exécute automatiquement, après la clôture américaine :

1. un régime **VERT / ORANGE / ROUGE** sur QQQ, SPY, Nasdaq et VIX ;
2. un scan des actions US ;
3. les filtres de tendance de type Minervini ;
4. un pivot mécanique ;
5. une entrée, un stop et une taille de position ;
6. un rapport mobile dans `docs/index.html` ;
7. une notification Telegram facultative.

## Hypothèses initiales

- Capital : 10 000 €
- Risque : 0,5 % par trade, soit environ 50 €
- Données quotidiennes ajustées
- Aucun ordre n'est envoyé au broker
- En régime rouge, aucun nouvel achat

Toutes ces valeurs sont modifiables dans `config.yml`.

## Mise en route

1. Créer un compte GitHub.
2. Créer un dépôt **public** vide.
3. Décompresser ce projet et envoyer tous les fichiers dans le dépôt.
4. Ouvrir l'onglet `Actions` et autoriser les workflows.
5. Dans `Settings > Pages`, choisir `Deploy from a branch`, branche `main`, dossier `/docs`.
6. Lancer une première fois `Daily market scan` avec `Run workflow`.
7. Ouvrir l'adresse GitHub Pages depuis le téléphone et l'ajouter à l'écran d'accueil.

## Telegram (facultatif)

Créer un bot avec BotFather, récupérer le token, puis ajouter dans GitHub :

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Chemin : `Settings > Secrets and variables > Actions`.

## Limites importantes

- `yfinance` utilise une interface Yahoo non contractuelle : elle peut changer ou subir des limites.
- Le pivot et la qualité de base sont encore mécaniques ; ce prototype ne reconnaît pas fiablement toutes les VCP.
- Les prix sont quotidiens, pas temps réel.
- Le calcul de taille utilise un capital en euros et des prix en dollars sans conversion FX : ce point doit être corrigé avant usage réel.
- Les résultats d'entreprise, le risque sectoriel et les positions déjà ouvertes ne sont pas encore intégrés.
- Ne pas saisir un ordre réel sans vérifier le graphique, le calendrier des résultats et le taux EUR/USD.

## Étapes suivantes indispensables avant argent réel

1. conversion EUR/USD ;
2. calendrier des résultats ;
3. journal des transactions réelles ;
4. prise en compte des positions ouvertes ;
5. analyse graphique et validation du pivot ;
6. tests historiques sans fuite d'information ;
7. mode parallèle sur au moins plusieurs dizaines de signaux.
