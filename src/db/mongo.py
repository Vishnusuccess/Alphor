from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

# ---------------- CONFIG ----------------
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGO_DB_NAME", "churn_system")

# ---------------- CLIENT ----------------
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=50000)
db = client[DB_NAME]

# ---------------- COLLECTIONS ----------------
data_col           = db["raw_data"]          # training / general dataset (10k)
client_data_col    = db["client_data"]       # client-uploaded data (120 rows)
model_col          = db["models"]             # GENERAL + CLIENT models
predictions_col    = db["predictions"]        # phase_0 predictions
feedback_col       = db["feedback"]           # PHASE_0 feedback
phase1_feedback_col = db["phase1_feedback"]  # PHASE_1 feedback
metrics_col        = db["metrics"]            # training + eval metrics
system_col         = db["system_state"]       # phase / autonomy flags
phase2_predictions_col = db["phase2_predictions"]
phase2_feedback_col    = db["phase2_feedback"]
phase2_actions_col     = db["phase2_actions"]
actions_col           = db["phase1_actions"]

# ---------------- INDEXES ----------------
data_col.create_index("ID")
client_data_col.create_index("ID")
predictions_col.create_index("ID")
feedback_col.create_index("ID")
phase1_feedback_col.create_index("ID")

phase2_predictions_col.create_index("ID")
phase2_predictions_col.create_index("timestamp")

phase2_actions_col.create_index("ID")
phase2_actions_col.create_index("phase")
phase2_actions_col.create_index("risk")
phase2_actions_col.create_index("auto_executed")
phase2_actions_col.create_index("timestamp")

phase2_feedback_col.create_index("ID")
phase2_feedback_col.create_index("risk_level")
phase2_feedback_col.create_index("auto_executed")
phase2_feedback_col.create_index("decision_correct")
phase2_feedback_col.create_index("timestamp")

model_col.create_index("model_type", unique=False)
system_col.create_index("key", unique=True)


