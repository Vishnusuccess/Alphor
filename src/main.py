from fastapi import FastAPI
from dotenv import load_dotenv
import os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
# ---------------- LOAD ENV ----------------
load_dotenv()

# ---------------- IMPORT ROUTERS ----------------
from routers.pre_phase_0 import router as pre_phase_0_router
from routers.phase_0 import router as phase_0_router
from routers.phase_1_loop import router as phase_1_router
from routers.phase_2_loop import router as phase_2_router

from dashboard.dashboard import router as dashboard_router
from dashboard.phase2_actions import router as phase2_actions_router  # ✅ Phase 2 APIs
from dashboard.system_control import router as system_router
from dashboard.phase2_enterprise import router as phase2_enterprise_router
from dashboard.retrain_status import router as retrain_router
# ---------------- INIT APP ----------------
app = FastAPI(title="Autonomous Churn Engine")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STATIC FILES ----------------
# Determine absolute path to src/dashboard_html
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "dashboard_html")  # src/dashboard_html

# Mount dashboard HTML files
app.mount("/dashboard_static", StaticFiles(directory=static_dir), name="dashboard_static")

# ---------------- INCLUDE ROUTERS ----------------
app.include_router(pre_phase_0_router, prefix="/pre_phase_0", tags=["PRE_PHASE_0"])
app.include_router(phase_0_router, prefix="/phase_0", tags=["PHASE_0"])

# Phase 1 routes:
# - keep existing /phase_1/... routes
app.include_router(phase_1_router, prefix="/phase_1", tags=["PHASE_1"])
# - add alias routes at root so frontend calling /feedback works
app.include_router(phase_1_router, prefix="", tags=["PHASE_1_ALIAS"])

app.include_router(phase_2_router, prefix="/phase_2", tags=["PHASE_2"])

app.include_router(dashboard_router, prefix="/dashboard")       # Phase 1 Dashboard APIs
app.include_router(phase2_actions_router, prefix="/dashboard")  # Phase 2 Dashboard APIs
app.include_router(system_router)
app.include_router(phase2_enterprise_router)
app.include_router(retrain_router)

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"message": "Go to /dashboard_static/index.html for the Phase 1 dashboard"}


