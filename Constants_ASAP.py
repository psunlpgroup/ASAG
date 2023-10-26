'''
https://huggingface.co/models
Most used:

bert-base-uncased
distilbert-base-uncased
roberta-base
google/electra-small-discriminator
YituTech/conv-bert-base
'''

# TRAIN_FILE_PATH = ['data/Col-STAT/debug.csv']
# TEST_FILE_PATH = ['data/Col-STAT/debug_test_set.csv']

TRAIN_FILE_PATH = ['./data/ASAP/4way/train_4way.csv']
TEST_FILE_PATH = ['./data/ASAP/4way/public_test_4way.csv']

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'

hyperparameters = dict(
    train_id="0904_ASAP_set56_test7",
    model_name="bert-base-uncased",
    num_labels = 4,
    max_length = 128,
    random_seed=23, # 23， 123
    data_split=0.2,
    lr=1e-5,
    epochs=10,
    weight_decay=0.01,
    GRADIENT_ACCUMULATION_STEPS=8,
    max_norm = 1, 
    WARMUP_STEPS=0.05,
    hidden_dropout_prob=0.2,
    # model
    hidden_dim=768, # 768
    mlp_hidden=128,
    )
# wandb config

config_dictionary = dict(
    #yaml=my_yaml_file,
    params=hyperparameters,
    )



q_text_dict = {
'5':'Starting with mRNA leaving the nucleus, list and describe four major steps involved in protein synthesis.',
'6':'List and describe three processes used by cells to control the movement of substances across the cell membrane.'
}

q_rubric_dict = {
    '5':['mRNA exits nucleus via nuclear pore.',
     'mRNA travels through the cytoplasm to the ribosome or enters the rough endoplasmic reticulum.',
     'mRNA bases are read in triplets called codons (by rRNA).',
     'tRNA carrying the complementary (U=A, C+G) anticodon recognizes the complementary codon of the mRNA.',
     'The corresponding amino acids on the other end of the tRNA are bonded to adjacent tRNA’s amino acids.',
     'A new corresponding amino acid is added to the tRNA.',
     'Amino acids are linked together to make a protein beginning with a START codon in the P site (initiation).',
     'Amino acids continue to be linked until a STOP codon is read on the mRNA in the A site (elongation and termination).',
    ],
    '6':['Selective permeability is used by the cell membrane to allow certain substances to move across.',
     'Passive transport occurs when substances move from an area of higher concentration to an area of lower concentration.',
     'Osmosis is the diffusion of water across the cell membrane.',
     'Facilitated diffusion occurs when the membrane controls the pathway for a particle to enter or leave a cell.',
     'Active transport occurs when a cell uses energy to move a substance across the cell membrane, and/or a substance moves from an area of low to high concentration, or against the concentration gradient.',
     'Pumps are used to move charged particles like sodium and potassium ions through membranes using energy and carrier proteins.',
     'Membrane-assisted transport occurs when the membrane of the vesicle fuses with the cell membrane forcing large molecules out of the cell as in exocytosis.',
     'Membrane-assisted transport occurs when molecules are engulfed by the cell membrane as in endocytosis.',
     'Membrane-assisted transport occurs when vesicles are formed around large molecules as in phagocytosis.',
     'Membrane-assisted transport occurs when vesicles are formed around liquid droplets as in pinocytosis.',
     'Protein channels or channel proteins allow for the movement of specific molecules or substances into or out of the cell.',
    ]
}