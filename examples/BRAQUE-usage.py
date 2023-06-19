import pandas as pd

import braque


# load necessary dbs

original_db = pd.read_csv("./original_db-toy.csv")
pos = pd.read_csv("./positions-toy.csv")
reference = pd.read_csv("./reference-toy.csv")


# start of the analysis

# minimal initialization of BRAQUE object
analysis = braque.BRAQUE(original_db=original_db, 
                         pos=pos,
                         reference=reference, 
                         # names from reference-toy.csv headers, minimal initialization requires 4 columns:
                         correspondence_column='Feature', 
                         naming_column='Official Name', 
                         interpretative_column='Significance', 
                         importance_column='Important')


# run the whole pipeline with standard parameters
analysis.run_pipeline()


### if you want to load data

# Once the analysis is completed, all the important steps will be stored in a
# minimal initialization for all premade steps loading results instead of computing them
analysis = braque.BRAQUE(original_db=original_db, 
                         pos=pos,
                         reference=reference, 
                         # names from reference-toy.csv headers, minimal initialization requires 4 columns:
                         correspondence_column='Feature', 
                         naming_column='Official Name', 
                         interpretative_column='Significance', 
                         importance_column='Important',
                         # it is possible to change other parameters when you load premade-stesp, 
                         # but the 'area' parameter must be identical to the one used previously
                         load_db=True, 
                         load_embed=True,
                         load_clusters=True)