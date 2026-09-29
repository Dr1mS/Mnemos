# Oubli — run qwen2.5:7b (local) : NON EFFECTUÉ cette nuit

**Ce qui s'est passé (30/09, 0 h 50 – 0 h 56).**

- La collecte commune s'est terminée normalement : 100 personas, 1 813 consignes
  d'oubli stockées sur 1 813. Le cache est intact
  (`%TEMP%/mnemos_forget_variants_sans_oubli_100.pkl`).
- Dès la 2ᵉ question, **Ollama a renvoyé des 500** : son processus de service
  (« runner ») ne répondait plus (`dial tcp 127.0.0.1:<port>: connectex … n'a pas
  répondu`). Journal conservé : `forget_variants_crash_0050.log`.
- Relance : même erreur dès l'appel d'essai.
- Déchargement du seul modèle qwen2.5:7b (sans toucher au reste d'Ollama) puis
  nouvel essai avec 9 Go de VRAM libres : **même erreur**. Le service Ollama est
  bloqué dans son ensemble ; seul un redémarrage complet d'Ollama le réparerait.
- **Ollama n'a pas été redémarré** : l'autre projet s'en sert aussi, et
  l'utilisateur dormait.

**Ce qui n'est pas touché.** La production (llama-server pour les embeddings,
aucun appel à Ollama) répond normalement : `/health` 200, llama-server 200.
Les runs distants (Ministral 14B, Nemotron 3 Ultra) tournent sur le même cache,
sans échec.

**Ce que ce run devait apporter, et comment le récupérer.** Son rôle principal
était le **contrôle de fidélité** : sur la moitié validation, les variantes A et
B simulées doivent retrouver les 26,7 % et 18,7 % mesurés par le vrai code avec
ce même répondeur. Pour le faire : redémarrer Ollama, puis

    PYTHONPATH=../Mnemos_AML_avant/src EMBED_BACKEND=llamacpp \
        .venv/Scripts/python.exe bench/bench_forget_variants.py

Le cache étant complet, il passe directement aux questions (~2 h).

Le script rattrape désormais toute erreur du répondeur (comptée dans le rapport)
et s'arrête proprement après 10 échecs consécutifs, au lieu de planter ou
d'écrire un rapport de réponses vides.
