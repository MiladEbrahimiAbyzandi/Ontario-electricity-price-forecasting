from pathlib import Path

#--Paths---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR/'artifacts'/'model'
SCALER_DIR = BASE_DIR/'artifacts'/'scalers'

#--model Hyperparameters-----------------------------------------------
INPUT_SIZE   = 9    
HIDDEN_SIZE  = 32
NUM_LAYERS   = 1
DROPOUT      = 0.2
SEQ_LENGTH   = 168 
HORIZONS     = [1, 2, 3]





