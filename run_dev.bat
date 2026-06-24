@echo off
title AlphaOptima Portfolio Suite Dev Environment
echo =========================================================
echo  AlphaOptima Portfolio Suite Dev Environment Launcher
echo =========================================================
echo.

:: 1. Launch FastAPI backend in a separate console window
echo [API] Launching FastAPI backend on http://127.0.0.1:8000 ...
start "AlphaOptima API Backend" cmd /k "cd backend && python server.py"

:: 2. Launch Vite React frontend in the current console window
echo [UI] Launching Vite React frontend on http://localhost:5173 ...
echo.
cd frontend
npm run dev
