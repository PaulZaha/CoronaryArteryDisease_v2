ordner_pfad = 'C:/Users/paulz/Documents/CoronaryArteryDisease_v2/Dataset/arcade/stenosis/val/masks/img'

import os

for dateiname in os.listdir(ordner_pfad):
    # Überprüfe, ob die Datei auf "_mask.png" endet
    if dateiname.endswith('_mask.png'):
        # Konstruiere den neuen Dateinamen, indem du "_mask" entfernst
        neuer_dateiname = dateiname.replace('_mask.png', '.png')
        # Konstruiere den vollen Pfad des alten und neuen Dateinamens
        alter_pfad = os.path.join(ordner_pfad, dateiname)
        neuer_pfad = os.path.join(ordner_pfad, neuer_dateiname)
        # Benenne die Datei um
        os.rename(alter_pfad, neuer_pfad)