## Modele początkowe

Modele początkowe definujemy są w `models.py` w `training_tuples`. Aby wytrenować modele początkowe, należy uruchomić `pretrain_models.py`.
Parametry treningu zdefiniowane są w `pretrain_flags.py`.

## Modele transferowe

Modele transferowe definujemy w `models.py` w `transfer_tuples`. Aby sprawdzić efektywność transferu, należy uruchomić `test_transfer.py`.
Parametry treningu zdefiniowane są w `transfer_flags.py`.
Przykładowa funkcja transferująca znajduje się w `transfer.py`.