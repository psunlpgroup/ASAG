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

TRAIN_FILE_PATH = ['./data/SemEval/5way/Beetle_train.csv']
TEST_FILE_PATH = ['./data/SemEval/5way/Beetle_test_UA.csv']

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'

hyperparameters = dict(
    train_id="0814_SFRN_beetle_UA5way",
    model_name="bert-base-cased",
    num_labels = 5,
    max_length = 128,
    random_seed=23, # 23， 123
    data_split=0.2,
    lr=1e-5,
    epochs=20,
    weight_decay=0.01,
    max_norm = 1, 
    WARMUP_STEPS=0.05,
    hidden_dropout_prob=0.2,
    GRADIENT_ACCUMULATION_STEPS = 8,
    # model
    hidden_dim=768, # 768
    mlp_hidden=512,
    )
# wandb config

config_dictionary = dict(
    #yaml=my_yaml_file,
    params=hyperparameters,
    )



q_text_dict = {
'PARALLEL_SWITCH_EXPLAIN_Q3':'What role does the path play in determining whether or not a switch affects a bulb?',
'VOLTAGE_DIFF_DISCUSS_1_Q':'What does a voltage reading of 0 tell you about the connection between a bulb terminal and a battery terminal?',
'BULB_C_VOLTAGE_EXPLAIN_WHY6':'Explain why you got a voltage reading of 0 for terminal 6 and the positive terminal.',
'TERMINAL_STATE_EXPLAIN_Q':'Why do you think those terminals and the negative battery terminal are in the same state?',
'BULB_ONLY_EXPLAIN_WHY4':'Explain why you got a voltage reading of 0 for terminal 1 and terminal 4.',
'PARALLEL_SWITCH_EXPLAIN_Q2':'Why was bulb C off when switch Z was open?',
'BULB_ONLY_EXPLAIN_WHY6':'Explain why you got a voltage reading of 1.5 for terminal 1 and terminal 6.',
'BURNED_BULB_LOCATE_EXPLAIN_Q':'Why does measuring voltage help you locate a burned out bulb? Try to answer in terms of electrical states, connections and/ or a gap.',
'PARALLEL_SWITCH_EXPLAIN_Q1':'Why was bulb A on when switch Y was open and switch Z was closed?',
'HYBRID_BURNED_OUT_EXPLAIN_Q3':'Explain your reasoning.',
'SWITCH_TABLE_EXPLAIN_Q1':'When switch X was closed and switch Y was open, why was bulb A on?',
'BULB_C_VOLTAGE_EXPLAIN_WHY1':'Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.',
'BULB_ONLY_EXPLAIN_WHY2':'Explain why you got a voltage reading of 0 for terminal 1 and terminal 2.',
'SWITCH_TABLE_EXPLAIN_Q3':'Under what circumstances will a switch affect a bulb?',
'SWITCH_OPEN_EXPLAIN_Q':'Why does an open switch impact a circuit?',
'SWITCH_TABLE_EXPLAIN_Q2':'When switch X was open and switch Y was closed, why were bulbs B and C on?',
'BULB_C_VOLTAGE_EXPLAIN_WHY2':'Explain why you got a voltage reading of 1.5 for terminal 2 and the positive terminal.',
'HYBRID_BURNED_OUT_EXPLAIN_Q1':'Explain your reasoning.',
'VOLTAGE_DIFF_DISCUSS_2_Q':'What does a voltage reading of 1.5 tell you about the connection between a bulb terminal and a battery terminal?',
'SHORT_CIRCUIT_EXPLAIN_Q_5':'Explain why circuit 5 is a short circuit.',
'DAMAGED_BULB_EXPLAIN_2_Q':'Why does a damaged bulb impact a circuit?',
'CONDITIONS_FOR_BULB_TO_LIGHT':'What are the conditions that are required to make a bulb light up?',
'SHORT_CIRCUIT_EXPLAIN_Q_4':'Explain why circuit 4 is not a short circuit.',
'SHORT_CIRCUIT_X_Q':'What do you think the red X means?',
'HYBRID_BURNED_OUT_WHY_Q3':'Under what circumstances will a damaged/ burned out bulb affect another bulb?',
'HYBRID_BURNED_OUT_WHY_Q2':'Why do both bulbs A and B stay on when bulb C is burned out?',
'SHORT_CIRCUIT_EXPLAIN_Q_2':'Explain why circuit 2 is not a short circuit.',
'VOLTAGE_GAP_EXPLAIN_WHY1':'Explain why you got a voltage reading of 1.5 for terminal 1 and the positive terminal.',
'VOLTAGE_GAP_EXPLAIN_WHY3':'Explain why you got a voltage reading of 1.5 for terminal 3 and the positive terminal.',
'VOLTAGE_GAP_EXPLAIN_WHY6':'Explain why you got a voltage reading of 0 for terminal 6 and the positive terminal.',
'VOLTAGE_DEFINE_Q':'What is voltage?',
'OTHER_TERMINAL_STATE_EXPLAIN_Q':'Why do you think the other terminals are being held in a different electrical state than that of the negative terminal?',
'VOLTAGE_GAP_EXPLAIN_WHY5':'Explain why you got a voltage reading of 0 for terminal 5 and the positive terminal.',
'VOLTAGE_GAP_EXPLAIN_WHY4':'Explain why you got a voltage reading of 0 for terminal 4 and the positive terminal.',
'VOLTAGE_INCOMPLETE_CIRCUIT_2_Q':'Why?',
'GIVE_CIRCUIT_TYPE_PARALLEL_EXPLAIN_Q2':'Explain your reasoning.',
'GIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q3':'Explain your reasoning.',
'GIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q2':'Explain your reasoning.',
'OPT1_EXPLAIN_Q2':'Why not?',
'BURNED_BULB_PARALLEL_EXPLAIN_Q3':'Explain your reasoning.',
'VOLTAGE_AND_GAP_DISCUSS_Q':'As you move the multimeter leads from one bulb terminal to the next, what does it mean when the voltage reading jumps from 0 to 1.5?',
'OPT2_EXPLAIN_Q':'Describe the paths in this diagram and explain how those paths account for the results.',
'BURNED_BULB_PARALLEL_EXPLAIN_Q2':'Explain your reasoning.',
'DAMAGED_BUILD_EXPLAIN_Q':'Explain your reasoning.',
'BURNED_BULB_PARALLEL_EXPLAIN_Q1':'Explain your reasoning.',
'BURNED_BULB_PARALLEL_WHY_Q':'Why didn^t bulbs A and C go out after bulb B burned out?',
'GIVE_CIRCUIT_TYPE_SERIES_EXPLAIN_Q':'Explain your reasoning.'
}

q_rubric_dict = {
'PARALLEL_SWITCH_EXPLAIN_Q3':['If a bulb and a switch are in the same path the switch affects the bulb', 'The switch and the bulb have to be in the same path'],
'VOLTAGE_DIFF_DISCUSS_1_Q':['the terminals are connected', 'the terminals are not separated by a gap', 'there is no gap between the terminals', 'The terminals are in the same state.'],
'BULB_C_VOLTAGE_EXPLAIN_WHY6':['Terminal 6 and the positive terminal are connected', 'Terminal 6 and the positive terminal are not separated by the gap', 'Terminal 6 and the positive terminal are not separated by the gap', 'terminal 6 is not connected to the negative battery terminal', 'There is no gap between terminal 6 and the positive terminal', 'terminal 6 is separated by a gap from the negative battery terminal', 'Terminal 6 and the positive battery terminal are in the same electrical state'],
'TERMINAL_STATE_EXPLAIN_Q':['Terminals 1, 2 and 3 are connected to the negative battery terminal', 'Terminals 1, 2 and 3 are not separated from the negative battery terminal by a gap', 'Terminals 1, 2 and 3 are not connected to the positive battery terminal', 'Terminals 1, 2 and 3 are separated from the positive battery terminal by a gap'],
'BULB_ONLY_EXPLAIN_WHY4':['Terminals 1 and 4 are connected', 'Terminals 1 and 4 are not separated by the gap', 'There is no gap between terminals 1 and 4', 'Terminals 1 and 4 are connected and in the same electrical state', 'Terminals 1 and 4 are in the same electrical state'],
'PARALLEL_SWITCH_EXPLAIN_Q2':['Bulb C was no longer in a closed path with the battery', 'There was no longer a closed path containing Bulb C and the battery', 'there is a path containing Z and C. The open switch Z creates a gap', 'there is a path containing Z and C. Opening of switch Z causes the path to be open', 'there is a path containing both Z and C', 'a gap', 'Bulb C was not in a closed path', 'Bulb C was in no closed path', 'There are no closed paths'],
'BULB_ONLY_EXPLAIN_WHY6':['Terminals 1 and 6 are separated by the gap', 'Terminals 1 and 6 are not connected', 'Terminals 1 and 6 are not connected and so in different electrical states', 'There is a gap between terminals 1 and 6', 'Terminals 1 and 6 are in different electrical states'],
'BURNED_BULB_LOCATE_EXPLAIN_Q':['Measuring voltage indicates the place where the electrical state changes due to a damaged bulb.', 'Measuring voltage indicates the place where the electrical state changes due to a gap.', 'Measuring voltage indicates whether two terminals are connected to each other.', 'Measuring voltage indicates whether two terminals are separated by a gap.', 'A zero voltage means that the terminals are connected.', 'A non-zero voltage means that the terminals are not connected.', 'A zero voltage means that the terminals are not separated by a gap.', 'A non-zero voltage means that the terminals are separated by a gap.', 'A non-zero voltage means that there is a damaged bulb.', 'A non-zero voltage means that there is a gap', 'A zero voltage means that there is no damaged bulb.', 'A zero voltage means that there is no gap', 'A non-zero voltage means that the terminals have different electrical states.', 'A zero voltage means that the terminals have the same electrical state.'],
'PARALLEL_SWITCH_EXPLAIN_Q1':['Bulb A is still contained in a closed path with the battery.', 'Bulb A is still contained in a closed path with the battery and switch Z.', 'There is no path containing both switch Y and bulb A', 'Switch Y and bulb A are not in the same closed path'],
'HYBRID_BURNED_OUT_EXPLAIN_Q3':['If C burns out, then A and B are still in a closed path with the battery.', 'If C burns out, then A and B are still in a closed path.'],
'SWITCH_TABLE_EXPLAIN_Q1':['Bulb A was still contained in the same closed path with the battery.', 'Bulb A is still contained in a closed path with the battery and switch X.', 'there is no path containing both switch Y and bulb A', 'Switch Y and bulb A are not in the same closed path'],
'BULB_C_VOLTAGE_EXPLAIN_WHY1':['Terminal 1 and the positive terminal are separated by the gap', 'Terminal 1 and the positive terminal are not connected', 'terminal 1 is connected to the negative battery terminal', 'terminal 1 is not separated from the negative battery terminal', 'Terminal 1 and the positive battery terminal are in different electrical states'],
'BULB_ONLY_EXPLAIN_WHY2':['Terminals 1 and 2 are connected', 'Terminals 1 and 2 are not separated by the gap', 'There is no gap between them', 'Terminals 1 and 2 are connected and in the same electrical state', 'There is no gap between terminals 1 and 2', 'Terminals 1 and 2 are in the same electrical state'],
'SWITCH_TABLE_EXPLAIN_Q3':['When the switch and the bulb are contained in the same path', 'When the switch and the bulb are contained in the same path'],
'SWITCH_OPEN_EXPLAIN_Q':['the open switch creates a gap', 'a gap', 'there is a gap in a circuit', 'the path is not closed', 'there is an incomplete circuit', 'there is an open path'],
'SWITCH_TABLE_EXPLAIN_Q2':['Bulb B and Bulb C were still contained in the same closed path with the battery.', 'no path containing both the switch X and either bulb B or bulb C', 'There is a path with B and C that does not include X'],
'BULB_C_VOLTAGE_EXPLAIN_WHY2':['Terminal 2 and the positive terminal are separated by the gap', 'Terminal 2 and the positive terminal are not connected', 'terminal 2 is connected to the negative battery terminal', 'terminal 2 is not separated from the negative battery terminal', 'Terminal 2 and the positive battery terminal are in different electrical states'],
'HYBRID_BURNED_OUT_EXPLAIN_Q1':['If bulb A burns out, B and C are no longer in a closed path with the battery', 'If bulb A burns out, there is no longer a closed path containing B, C and the battery', 'If bulb A burns out, neither B nor C is in a closed path'],
'VOLTAGE_DIFF_DISCUSS_2_Q':['the terminals are not connected', 'the terminals are separated by a gap', 'The terminals are separated.', 'There is a gap.', 'The terminals are in different electrical states.'],
'SHORT_CIRCUIT_EXPLAIN_Q_5':['the battery is contained in a path in which there is no bulb', 'The battery is contained in a path which does not contain any other components', 'The bulb is not in the closed path containing the battery', 'the battery is contained in a closed path in which there is no bulb', 'The battery is contained in a closed path which does not contain any other components'],
'DAMAGED_BULB_EXPLAIN_2_Q':['a damaged bulb creates a gap', 'there is a gap in the circuit', 'there is an open path', 'there is no closed path', 'there is an incomplete circuit'],
'CONDITIONS_FOR_BULB_TO_LIGHT':['there is a closed path containing both the bulb and a battery', 'the bulb and the battery are in a closed path', 'the circuit has a closed path', 'there is a complete circuit', 'One terminal of the bulb has to be connected to one terminal of the battery, and the other terminal of the bulb has to be connected to the different terminal of the battery.'],
'SHORT_CIRCUIT_EXPLAIN_Q_4':['Circuit 4 has no closed paths', 'The battery in 4 is not in a closed path', 'There is no closed path', 'the battery is in an open path'],
'SHORT_CIRCUIT_X_Q':['the battery is damaged', 'there is a short circuit'],
'HYBRID_BURNED_OUT_WHY_Q3':['If they are in the same path the burned out bulb affects the other bulb.', 'It depends on whether or not they are in the same path', 'The damaged bulb and the other bulb must be contained in the same path'],
'HYBRID_BURNED_OUT_WHY_Q2':['If C burns out then A and B are still in a closed path with the battery', 'There is a closed path with A and B that C is not in.', 'If C is damaged, A and B are still in a closed path', 'A and B have their own path', 'There is still a closed path'],
'SHORT_CIRCUIT_EXPLAIN_Q_2':['The battery in 2 is not in a closed path', 'there is no closed path containing the battery', 'the battery is in an open path'],
'VOLTAGE_GAP_EXPLAIN_WHY1':['Terminal 1 and the positive terminal are separated by the gap', 'Terminal 1 and the positive terminal are not connected', 'terminal 1 is connected to the negative battery terminal', 'terminal 1 is not separated from the negative battery terminal', 'terminal 1 and the positive battery terminal are in different electrical states'],
'VOLTAGE_GAP_EXPLAIN_WHY3':['Terminal 3 and the positive terminal are separated by the gap', 'Terminal 3 and the positive terminal are not connected', 'terminal 3 is connected to the negative battery terminal', 'terminal 3 is not separated from the negative battery terminal', 'Terminal 3 and the positive battery terminal are in different electrical states'],
'VOLTAGE_GAP_EXPLAIN_WHY6':['Terminal 6 and the positive terminal are connected', 'Terminal 6 and the positive terminal are not separated by the gap', 'There is no gap between terminal 6 and the positive terminal', 'terminal 6 is not connected to the negative battery terminal', 'terminal 6 is separated by a gap from the negative battery terminal', 'Terminal 6 and the positive battery terminal are in the same electrical state'],
'VOLTAGE_DEFINE_Q':['Voltage is the difference in electrical states between two terminals','Voltage is the difference in electrical states between two terminals'],
'OTHER_TERMINAL_STATE_EXPLAIN_Q':['Terminals 4, 5 and 6 are not connected to the negative battery terminal', 'Terminals 4, 5 and 6 are separated from the negative battery terminal by a gap', 'Terminals 4, 5 and 6 are connected to the positive battery terminal', 'Terminals 4, 5 and 6 are not separated from the positive battery terminal by a gap'],
'VOLTAGE_GAP_EXPLAIN_WHY5':['Terminal 5 and the positive terminal are connected', 'Terminal 5 and the positive terminal are not separated by the gap', 'There is no gap between terminal 5 and the positive terminal', 'terminal 5 is not connected to the negative battery terminal', 'terminal 5 is separated by a gap from the negative battery terminal', 'Terminal 5 and the positive battery terminal are in the same electrical state'],
'VOLTAGE_GAP_EXPLAIN_WHY4':['Terminal 4 and the positive terminal are connected', 'Terminal 4 and the positive terminal are not separated by the gap', 'There is no gap between terminal 4 and the positive terminal', 'terminal 4 is not connected to the negative battery terminal', 'terminal 4 is separated by a gap from the negative battery terminal', 'Terminal 4 and the positive battery terminal are in the same electrical state'],
'VOLTAGE_INCOMPLETE_CIRCUIT_2_Q':['A battery uses a chemical reaction to maintain different electrical states at the terminals','A battery uses a chemical reaction to maintain different electrical states at the terminals'],
'GIVE_CIRCUIT_TYPE_PARALLEL_EXPLAIN_Q2':['A and C are in different paths with the battery', 'A and C are not in the same path with the battery', 'A and C are in different paths', 'A and C are not in the same path'],
'GIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q3':['B and C are in the same path with the battery', 'B and C are in the same path'],
'GIVE_CIRCUIT_TYPE_HYBRID_EXPLAIN_Q2':['A and C are in different paths with the battery', 'A and C are not in the same path with the battery', 'A and C are in different paths', 'A and C are not in the same path'],
'OPT1_EXPLAIN_Q2':['there is still a closed path containing Switch Z, Switch X, Bulbs A and C and the battery', 'there is a path containing A, C, Z and the battery', 'there is a path containing A, C, Z and X', 'There is a path containing A', 'Bulb A is still in a closed path with the battery'],
'BURNED_BULB_PARALLEL_EXPLAIN_Q3':['bulbs A and B are still in closed paths with the battery', 'bulbs A and B are in a closed path', 'each bulb is in its own path', 'Bulb C is in a separate path'],
'VOLTAGE_AND_GAP_DISCUSS_Q':['the bulb is damaged', 'the terminals are not connected', 'there is a gap', 'the terminals are separated by a gap'],
'OPT2_EXPLAIN_Q':["Bulb A is in a path which does not contain B and C, so bulbs B and C don't affect it. Bulbs B and C are in the same path. They affect each other, but Bulb A doesn't affect them.", "Bulb A is in a path which does not contain B and C and isn't affected by B or C. B and C are in the same path and affect each other.", 'there is a path containing A and a different path containing B and C'],
'BURNED_BULB_PARALLEL_EXPLAIN_Q2':['Bulbs A and C are still in closed paths with the battery', 'Bulbs A and C are in a closed path', 'each bulb is in its own path', 'Bulb B is in a separate path'],
'DAMAGED_BUILD_EXPLAIN_Q':['bulb B creates a gap', 'a gap', 'there is a gap in a circuit', 'there is an open path', 'there is an incomplete circuit', 'there is no closed path'],
'BURNED_BULB_PARALLEL_EXPLAIN_Q1':['bulbs B and C are still in closed paths with the battery', 'If bulb A burns out, bulbs B and C are still in a closed path', 'each bulb is in its own path'],
'BURNED_BULB_PARALLEL_WHY_Q':['Bulbs A and C are still contained in closed paths with the battery', 'Bulbs A and C are still in closed paths', 'B is not in their paths', 'A, B and C are in different paths'],
'GIVE_CIRCUIT_TYPE_SERIES_EXPLAIN_Q':['A and C are in the same closed path', 'A and C are in the same path']
} 
